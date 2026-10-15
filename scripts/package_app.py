#!/usr/bin/env python3
"""Build a self-contained local macOS app using a relocatable Python distribution."""
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'dist' / 'RoboVerse Studio.app'
STAGING = ROOT / 'dist' / '.RoboVerse Studio.build.app'


def main():
    if sys.platform != 'darwin':
        raise SystemExit('This app must be packaged on macOS.')
    commands = subprocess.check_output(['ps', '-axo', 'comm='], text=True).splitlines()
    if str(DEST / 'Contents/MacOS/RoboVerseStudio') in (line.strip() for line in commands):
        raise SystemExit('Quit RoboVerse Studio before replacing its app bundle, then run make app again.')
    base = Path(sys.base_prefix)
    # Framework/system Pythons need a dedicated relocatable runtime; never silently ship broken symlinks.
    if 'Python.framework' in str(base) or str(base).startswith('/System/'):
        raise SystemExit('Use a relocatable Python (for example uv-managed CPython) to create .venv, then rerun.')
    subprocess.run(['swift', 'build', '-c', 'release'], cwd=ROOT, check=True)
    products = Path(subprocess.check_output(['swift', 'build', '-c', 'release', '--show-bin-path'], cwd=ROOT, text=True).strip())
    STAGING.parent.mkdir(exist_ok=True)
    if STAGING.exists(): shutil.rmtree(STAGING)
    contents = STAGING / 'Contents'
    resources = contents / 'Resources'
    (contents / 'MacOS').mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(products / 'RoboVerseStudio', contents / 'MacOS' / 'RoboVerseStudio')
    shutil.copytree(base, resources / 'runtime', symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    packages = Path(sys.prefix) / 'lib' / version / 'site-packages'
    shutil.copytree(packages, resources / 'runtime' / 'lib' / version / 'site-packages', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'python', resources / 'python', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'models', resources / 'models')
    shutil.copytree(ROOT / 'examples', resources / 'examples',
                    ignore=shutil.ignore_patterns('.venv', '__pycache__', '*.pyc', 'projects', 'assets', 'MUJOCO_LOG.TXT'))
    if (ROOT / 'examples' / 'projects').is_dir():
        shutil.copytree(ROOT / 'examples' / 'projects', resources / 'projects')
    for name in ('README.md', 'README.zh-CN.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(ROOT / name, resources / name)
    shutil.copytree(ROOT / 'docs', resources / 'docs', dirs_exist_ok=True)
    shutil.copy2(ROOT / 'Sources/MjbatchStudio/Resources/translations.json', resources / 'translations.json')
    info = {
        'CFBundleIdentifier': 'com.sinpoce.roboverse.studio',
        'CFBundleName': 'RoboVerse Studio', 'CFBundleDisplayName': 'RoboVerse Studio',
        'CFBundleExecutable': 'RoboVerseStudio', 'CFBundlePackageType': 'APPL',
        'CFBundleShortVersionString': '0.4.0', 'CFBundleVersion': '4',
        'CFBundleDevelopmentRegion': 'en', 'CFBundleLocalizations': ['en', 'zh-Hans'],
        'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True,
        'NSPrincipalClass': 'NSApplication', 'CFBundleIconFile': 'RoboVerseStudio.icns',
        'LSApplicationCategoryType': 'public.app-category.developer-tools',
        'CFBundleDocumentTypes': [
            {'CFBundleTypeName': 'RoboVerse Studio Project', 'CFBundleTypeRole': 'Editor',
             'LSHandlerRank': 'Owner', 'LSItemContentTypes': ['com.sinpoce.roboverse.studio.project']},
            {'CFBundleTypeName': 'RoboVerse Studio Recording', 'CFBundleTypeRole': 'Viewer',
             'LSHandlerRank': 'Owner', 'LSItemContentTypes': ['com.sinpoce.roboverse.studio.recording']},
        ],
        'UTExportedTypeDeclarations': [
            {'UTTypeIdentifier': 'com.sinpoce.roboverse.studio.project', 'UTTypeDescription': 'RoboVerse Studio Project',
             'UTTypeConformsTo': ['public.data'], 'UTTypeTagSpecification': {'public.filename-extension': ['mjstudio']}},
            {'UTTypeIdentifier': 'com.sinpoce.roboverse.studio.recording', 'UTTypeDescription': 'RoboVerse Studio Recording',
             'UTTypeConformsTo': ['public.data'], 'UTTypeTagSpecification': {'public.filename-extension': ['mjrec']}},
        ],
    }
    subprocess.run(['swift', str(ROOT/'scripts/make_icon.swift'), str(resources/'RoboVerseStudio.icns')], check=True)
    with (contents / 'Info.plist').open('wb') as file: plistlib.dump(info, file)
    runtime = resources / 'runtime' / 'bin' / 'python3'
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    subprocess.run([str(runtime), '-c', 'import mujoco, numpy, PIL, websockets, torch, yaml; print("Bundled runtime:", mujoco.__version__, "training:", torch.__version__)'], cwd='/tmp', env=env, check=True)
    # Ad hoc signing supports local use. Developer ID signing/notarization is a separate release step.
    magic = {b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}
    for path in sorted(contents.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as file: signature = file.read(4)
            if signature in magic:
                subprocess.run(['codesign', '--force', '--sign', '-', str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['codesign', '--force', '--sign', '-', str(STAGING)], check=True)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(STAGING)], check=True)
    if DEST.exists():
        previous = DEST.with_name('RoboVerse Studio.previous.app')
        if previous.exists(): shutil.rmtree(previous)
        DEST.rename(previous)
    STAGING.rename(DEST)
    print(f'Built: {DEST}')


if __name__ == '__main__': main()
