import os
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    cythonize = None


package_name = "pixtreme"
source_dir_name = "pixtreme_source"
base_dir = os.path.join(os.path.dirname(__file__), "src")
source_dir = os.path.join(base_dir, source_dir_name)
dest_dir = os.path.join(base_dir, package_name)
os.makedirs(dest_dir, exist_ok=True)


class BuildExtWithStubs(build_ext):
    def run(self):
        # build extensions
        super().run()

        procs = []
        for ext in self.extensions:
            assert isinstance(ext, Extension), f"Extension must be of type Extension, got {type(ext)}"

            # get the extension name
            fullname = ext.name  # e.g. pixtreme.aces.aces_transform
            filename = self.get_ext_filename(fullname)  # e.g. pixtreme/aces/aces_transform.cp312-win_amd64.pyd
            target_path = os.path.join(self.build_lib, filename)

            if not os.path.exists(target_path):
                print(f"[stubgen] Cannot find: {target_path}", file=sys.stderr)
                continue

            # set outdir
            if getattr(self, "inplace", False):
                print(f"[stubgen] Inplace build: {self.inplace}")

                outdir = "src"
                # create stub file
                cmd = [sys.executable, "-m", "pybind11_stubgen", "--output", outdir, fullname]
                print(f"[stubgen] Generating stubs for {fullname} → {outdir}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                procs.append(proc)

            else:
                print(f"[stubgen] Not inplace build: {self.inplace}")
                package_name = fullname.split(".")[0]  # e.g. pixtreme

                source = ext.sources[0]  # e.g. src/pixtreme/aces/aces_transform.cpp
                _sources = source.split("/")
                source = os.path.join(*_sources[2:])  # e.g. aces/aces_transform.cpp
                print(f"[stubgen] Source: {source}")

                source_stub_path = os.path.join("src", package_name, source).replace(".cpp", ".pyi")
                dest_stub_path = os.path.join(self.build_lib, package_name, source).replace(".cpp", ".pyi")
                print(f"[stubgen] Source stub path: {source_stub_path}")
                print(f"[stubgen] Destination stub path: {dest_stub_path}")

                if os.path.exists(source_stub_path):
                    print(f"[stubgen] Source stub exists: {source_stub_path}")
                    os.makedirs(os.path.dirname(dest_stub_path), exist_ok=True)
                    shutil.copyfile(source_stub_path, dest_stub_path)

        # wait for all processes to finish
        for proc in procs:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"[stubgen] Error occurred: {stderr.decode()}", file=sys.stderr)
            else:
                print(f"[stubgen] Success: {stdout.decode()}")


def get_python_includes() -> list[str]:
    includes = [
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        os.path.join(sysconfig.get_path("include"), "include"),
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Include"),
    ]
    if hasattr(sys, "base_prefix"):
        includes.extend(
            [
                os.path.join(sys.base_prefix, "include"),
                os.path.join(sys.base_prefix, "Include"),
            ]
        )
    return list(set(filter(os.path.exists, includes)))


include_dirs = get_python_includes() + ["src"]

if sys.platform == "win32":
    print("Building for win32 platform.")
    extra_compile_args = [
        "/utf-8",
        "/O2",
        # "/MP8",
        "/Oi",
        "/GL",
        "/Gy",
        "/GF",
        "/Gw",
        "/GS-",
        "/DNDEBUG",
    ]
    extra_link_args = [
        "/LTCG",
        "/OPT:REF",
        "/OPT:ICF",
    ]
else:
    extra_compile_args = [
        "-O3",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
        "-fno-stack-protector",
        "-fvisibility=hidden",
        "-fomit-frame-pointer",
        "-msse4.2",
    ]
    extra_link_args = [
        "-s",
        "-Wl,--strip-all",
        "-Wl,-z,now",
        "-Wl,-z,relro",
    ]

sources = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".py") and not file == "__init__.py":
            print(f"Processing file: {file} in {root}")
            rel_path = os.path.relpath(os.path.join(root, file), source_dir)
            sources.append(rel_path)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        elif file == "__init__.py" or file == "py.typed":
            rel_path = os.path.relpath(os.path.join(root, file), source_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(os.path.join(root, file), dest_path)
        # elif file.endswith(".cpp") or file.endswith(".h"):
        #    os.remove(os.path.join(root, file))


third_party_dir = os.path.join(os.path.dirname(__file__), "third_party")
for root, _, files in os.walk(third_party_dir):
    for file in files:
        if file.endswith(".cube"):
            print(f"Processing LUT file: {file} in {root}")
            lut_dir = os.path.join(dest_dir, "color", "lut")
            os.makedirs(lut_dir, exist_ok=True)
            dest_path = os.path.join(lut_dir, file)
            print(f"Copying to: {dest_path}")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(os.path.join(root, file), dest_path)
        else:
            print(f"Skipping third-party file: {file} in {root}")


print("sources:", sources)

source_dicts = []
for _source in sources:
    # source_path = os.path.join(source_dir, _source)
    source_path = os.path.join("src", source_dir_name, _source).replace("\\", "/")
    module_name = os.path.splitext(_source)[0].replace("/", ".").replace("\\", ".")
    source_dicts.append({module_name: source_path})

extensions = []
# if "build_ext" in sys.argv or "build" in sys.argv:
for source in source_dicts:
    for module_name, source_path in source.items():
        extensions.append(
            Extension(
                name=f"{package_name}.{module_name}",
                sources=[source_path],
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                depends=[],
                language="c++",
            )
        )


ext_modules = []
if USE_CYTHON and extensions and cythonize:
    print("Cython is available!, using Cython extensions.")
    ext_modules = cythonize(
        extensions,
        # nthreads=8,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "profile": False,
            "linetrace": False,
            "embedsignature": False,
        },
        compile_time_env={
            "SECURE_BUILD": True,
            "DEBUG": False,
        },
    )
else:
    print("Cython is not available, using pure C++ extensions.")


setup(
    name=package_name,
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=[f"{source_dir_name}", f"{source_dir_name}.*"]),
    package_data={
        package_name: [
            "__init__.py",
            "*.pyd",
            "*.so",
            "*.pyi",
            "*.cube",
            "py.typed",
        ],
    },
    cmdclass={"build_ext": BuildExtWithStubs},
    # cmdclass={"build_ext": build_ext},
    options={"build_ext": {"parallel": 32}},
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False,
)
