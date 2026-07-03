# Changelog

## [2.9.0](https://github.com/haexhub/fwbg/compare/v2.8.1...v2.9.0) (2026-07-03)


### Features

* **workspace:** seed default presets into fresh workspaces at startup ([75e3642](https://github.com/haexhub/fwbg/commit/75e3642064d8625eb1fcdb34bbd9ab9acfd427ab))
* **workspace:** seed default presets into fresh workspaces at startup ([dc1ed5a](https://github.com/haexhub/fwbg/commit/dc1ed5abddcfe0565021ad6916d0d656977f3078))

## [2.8.1](https://github.com/haexhub/fwbg/compare/v2.8.0...v2.8.1) (2026-07-02)


### Bug Fixes

* **docker:** install fwbg-premium from local path ([fd19360](https://github.com/haexhub/fwbg/commit/fd19360c6f63f95ba6a2df92e41f8645a1f76dce))
* **docker:** install fwbg-premium from local path ([98f3a63](https://github.com/haexhub/fwbg/commit/98f3a63aa19844d3b6588d3fa2c1d9908ff0c7b0))

## [2.8.0](https://github.com/haexhub/fwbg/compare/v2.7.1...v2.8.0) (2026-07-02)


### Features

* **premium:** add fwbg-premium as first-class dependency ([#37](https://github.com/haexhub/fwbg/issues/37)) ([e574c4a](https://github.com/haexhub/fwbg/commit/e574c4a69c3a512dc850e5269062453e4b4ae5e8))

## [2.7.1](https://github.com/haexhub/fwbg/compare/v2.7.0...v2.7.1) (2026-07-02)


### Bug Fixes

* harden bot, broker adapter and API layer (full code review) ([#32](https://github.com/haexhub/fwbg/issues/32)) ([e5a340a](https://github.com/haexhub/fwbg/commit/e5a340a85c597cd1a0b1371c660276c2fc3d0d6f))

## [2.7.0](https://github.com/haexhub/fwbg/compare/v2.6.0...v2.7.0) (2026-07-02)


### Features

* **api:** asset-registry endpoints as single source of truth ([#27](https://github.com/haexhub/fwbg/issues/27)) ([a8cd1f1](https://github.com/haexhub/fwbg/commit/a8cd1f1e8adbf5b7c2321407750b3cc5cd9ea0dd))
* **api:** on-demand data provisioning (POST /api/data/ensure) ([2c25724](https://github.com/haexhub/fwbg/commit/2c257247aec2bd511be2bf41f453a82c1354a4b5))
* **api:** on-demand data provisioning endpoint (POST /api/data/ensure) ([318e375](https://github.com/haexhub/fwbg/commit/318e37587da2e5835349f908d9df3d8f87cb748e))


### Bug Fixes

* **compose:** point api & bot data dir at the shared fwbg-data volume ([#26](https://github.com/haexhub/fwbg/issues/26)) ([e8d19f1](https://github.com/haexhub/fwbg/commit/e8d19f199dfe99b511d33765f62076d4bbca8809))

## [2.6.0](https://github.com/haexhub/fwbg/compare/v2.5.0...v2.6.0) (2026-06-30)


### Features

* DataSourceAdapter list_assets + GET /api/datasources/assets ([#21](https://github.com/haexhub/fwbg/issues/21)) ([762d6ef](https://github.com/haexhub/fwbg/commit/762d6ef075c7e87e1d07b8b0553dddbebc18f72f))


### Bug Fixes

* **docker:** run the REST API as its own service ([36d6e04](https://github.com/haexhub/fwbg/commit/36d6e04766bba12bef8230cd86b02837a96d213b))
* **docker:** run the REST API as its own service ([58e141f](https://github.com/haexhub/fwbg/commit/58e141f2762f1bab04332152d8b3d991d9c641ad))

## [2.5.0](https://github.com/haexhub/fwbg/compare/v2.4.2...v2.5.0) (2026-06-30)


### Features

* Dukascopy historical data adapter ([bfb67c3](https://github.com/haexhub/fwbg/commit/bfb67c38d809865999d5c2df3b473f8a37d310c2))
* Dukascopy historical data adapter ([f68efc0](https://github.com/haexhub/fwbg/commit/f68efc0adccb68d43245249c180328bc0e03d89d))
* **dukascopy:** instrument catalogue + data-driven backtest spreads ([#16](https://github.com/haexhub/fwbg/issues/16)) ([d5df4d9](https://github.com/haexhub/fwbg/commit/d5df4d92c6790f3646376990ba29177a97a37bfb))

## [2.4.2](https://github.com/haexhub/fwbg/compare/v2.4.1...v2.4.2) (2026-06-28)


### Bug Fixes

* remove invalid extra-files type from release-please config ([408e944](https://github.com/haexhub/fwbg/commit/408e94426d3aa13192de0b1c11e923cf1404778e))
* wire FWBG_AGENTS_API_URL through to dashboard container ([9873dcd](https://github.com/haexhub/fwbg/commit/9873dcd1f7d3298b9e5ef1667575e8f4494bf46c))


### Documentation

* **M4b:** mark fwbg-agents M4b plan done — Researcher search resilience + fan-out ([c6ba0f2](https://github.com/haexhub/fwbg/commit/c6ba0f29bfb6080f8e4cf0186b3bbab9c89f74bf))
