# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

See the [roadmap](https://github.com/niessner/Opt/blob/master/ROADMAP.md) for near-future changes.

## [Unreleased]

## [0.2.2] - 2017-11-29
### Added
- Higher verbosity levels, along with documentation

### Fixed 
- Robustness to divide-by-zero in pure GN solver.

## [0.2.1] - 2017-10-25
### Added
- Roadmap
- Partially-automated regression test script.
- Verbose-mode GPU memory usage reporting
- Improved documentation 
- Improved error messages

### Fixed 
- [#91](https://github.com/niessner/Opt/issues/91): Graph energies with no unknown energies produce erroneous results

## [0.2.0] - 2017-10-22
### Added
- Changelog
- Semantic version number
- threadsPerBlock initialization parameter. Globally sets #threads/block
- Auto-detection of native double-precision floating point atomics

### Fixed 
- Longstanding code emission bug on graph domains

## [0.1.1] - 2017-07-05
### Added
- Simple tests/

### Fixed 
- Memory leaks

[Unreleased]: https://github.com/niessner/Opt/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/niessner/Opt/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/niessner/Opt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/niessner/Opt/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/niessner/Opt/v0.1.0...v0.1.1
