# Changelog

## [2.17.0](https://github.com/haexhub/fwbg/compare/v2.16.0...v2.17.0) (2026-07-16)


### Features

* **bot:** record signal_price and assumed_spread in trade telemetry ([b9a251b](https://github.com/haexhub/fwbg/commit/b9a251b12010003178afca117acca253bca47ccd))
* **bot:** record signal_price and assumed_spread in trade telemetry ([675ffdf](https://github.com/haexhub/fwbg/commit/675ffdfbf9c322052f7442d7889b9156abb19483))
* **data:** data-quality report at Dukascopy download ([131a532](https://github.com/haexhub/fwbg/commit/131a5329ec9c13b830075762e8d5ce7ea85c3540))
* **data:** write data-quality report at Dukascopy download ([f852890](https://github.com/haexhub/fwbg/commit/f852890079cf8710d1a57f954a989618aac322ec))

## [2.16.0](https://github.com/haexhub/fwbg/compare/v2.15.0...v2.16.0) (2026-07-15)


### Features

* **runs:** expose run duration in list endpoint ([853a420](https://github.com/haexhub/fwbg/commit/853a4209d35e8eaf2902cf36603e33a06c5797d9))
* **runs:** expose run duration in list endpoint ([ed3283a](https://github.com/haexhub/fwbg/commit/ed3283aee5364b7be18a6d616a9471830199fdfb))

## [2.15.0](https://github.com/haexhub/fwbg/compare/v2.14.0...v2.15.0) (2026-07-15)


### Features

* **optimization:** causal risk calibration for reporting + time-based equity replay ([0c01be5](https://github.com/haexhub/fwbg/commit/0c01be5df5e8510e20c2aa0c5b81d0dab22d45b5))
* **simulation:** attach vol_regime/trend_regime labels to trade dicts (Plan 010 WP5) ([f736edd](https://github.com/haexhub/fwbg/commit/f736eddf5c2eabb64c22c73a99b767b6d77d42e8))


### Bug Fixes

* **api:** drop frozen 'forexsb' source defaults, resolve at runtime ([8d4bca2](https://github.com/haexhub/fwbg/commit/8d4bca2b9df8603cef339561b946e1b6f781fdc3))
* phantom SL wins (sl_level validation) + causal risk reporting + time-based equity replay ([78b48d0](https://github.com/haexhub/fwbg/commit/78b48d04fb8d780979207988a1fd6ddc4a2f5e05))
* **simulation:** reject invalid sl_level suffixes and guard wrong-side SL levels ([7b3268a](https://github.com/haexhub/fwbg/commit/7b3268ac048967af5f43a7a7fb5a64041b5a62a0))

## [2.14.0](https://github.com/haexhub/fwbg/compare/v2.13.0...v2.14.0) (2026-07-13)


### Features

* **runs:** backtest date window + cost multiplier (fwbg-agents Plan 009 WP4) ([461b46c](https://github.com/haexhub/fwbg/commit/461b46c137852235de976000430c065c2e3f2849))

## [2.13.0](https://github.com/haexhub/fwbg/compare/v2.12.0...v2.13.0) (2026-07-12)


### Features

* plugin source & spec endpoints, stability and data fixes ([3533711](https://github.com/haexhub/fwbg/commit/35337119869121c77067ffe48a603f40a5a396cd))
* **plugins:** namespace filter in GET /api/plugins + agent-authored MCP resolution ([fe7aa1a](https://github.com/haexhub/fwbg/commit/fe7aa1a484ac614149bdee8d5d7b6aec5418f661))
* **plugins:** Plan 007 — namespace filter + agent-authored MCP resolution ([3a687a6](https://github.com/haexhub/fwbg/commit/3a687a6cf16333eaecc82d6244f084f868f283e8))

## [2.12.0](https://github.com/haexhub/fwbg/compare/v2.11.0...v2.12.0) (2026-07-10)


### Features

* **broker:** reject entry orders without a stop-loss (deterministic gate) ([90d98f2](https://github.com/haexhub/fwbg/commit/90d98f21545151a3c823d0e41e9c30f189dbf8e3))
* **broker:** reject entry orders without a stop-loss (deterministic gate) ([1ad92d3](https://github.com/haexhub/fwbg/commit/1ad92d3a383bb30b409631ecce65f9cc73b95c22))
* mandatory stop-loss gate + v2.11.0 sync ([62805de](https://github.com/haexhub/fwbg/commit/62805de5749f209f3f679f22dfce51c662fd95a9))


### Bug Fixes

* **broker:** add limit_distance range validation to packages adapter ([986c658](https://github.com/haexhub/fwbg/commit/986c658a2c1ab51f02b032eaa6826cf40ec26a7e))
* **broker:** address code review findings for PR [#97](https://github.com/haexhub/fwbg/issues/97) ([4b9c04e](https://github.com/haexhub/fwbg/commit/4b9c04e665b855fb1c590dbd13d6523cc4d3b929))

## [2.11.0](https://github.com/haexhub/fwbg/compare/v2.10.8...v2.11.0) (2026-07-09)


### Features

* **api:** POST /api/plugins — register agent-authored plugins at runtime ([bdf1ec2](https://github.com/haexhub/fwbg/commit/bdf1ec2d30638afc76314e9336e0c0a11c981ea8))
* **api:** POST /api/plugins — register agent-authored plugins at runtime ([e6aaf2b](https://github.com/haexhub/fwbg/commit/e6aaf2b5fdcf8872b7533b72bf3bcbf850622239))
* **api:** serve plugin source via /api/plugins/{fqn}/source ([5255f32](https://github.com/haexhub/fwbg/commit/5255f32d0bdcfbcbbef212487e915985e4abe85d))
* **api:** serve plugin source via GET /api/plugins/{fqn}/source ([38f8fba](https://github.com/haexhub/fwbg/commit/38f8fba80e2825557147682816aceb87e455ffd8))
* **plugins:** speckit spec.md for all plugins + GET /api/plugins/{fqn}/spec ([6c9e294](https://github.com/haexhub/fwbg/commit/6c9e294a0050792b3f1ad1cd77de1091d6b0e086))
* **plugins:** speckit specs for all plugins + spec endpoint ([1cb577b](https://github.com/haexhub/fwbg/commit/1cb577b6a0525c8514f348ca23f0a7618ac10601))


### Bug Fixes

* **review:** address all valid PR [#92](https://github.com/haexhub/fwbg/issues/92) code review findings ([e55bee0](https://github.com/haexhub/fwbg/commit/e55bee01943ed6975a84808482fd8a45b4a3f722))

## [2.10.8](https://github.com/haexhub/fwbg/compare/v2.10.7...v2.10.8) (2026-07-08)


### Bug Fixes

* **data:** stop full-history re-download stall blocking agent runs ([443c3d5](https://github.com/haexhub/fwbg/commit/443c3d50dce84ae8b6915767fd38d781ffdc3168))
* **data:** stop full-history re-download stall blocking agent runs ([bf36da1](https://github.com/haexhub/fwbg/commit/bf36da1f855b42173526f1a9dfc0495f203f458a))

## [2.10.7](https://github.com/haexhub/fwbg/compare/v2.10.6...v2.10.7) (2026-07-06)


### Bug Fixes

* **backtest:** correct metrics under Python 3.14 forkserver workers ([6ff8eae](https://github.com/haexhub/fwbg/commit/6ff8eae38d867cb42d5cd6f90b7e8dfd92db2bea))
* **backtest:** correct metrics under Python 3.14 forkserver workers ([9bffb81](https://github.com/haexhub/fwbg/commit/9bffb818553651b6ccf1669183aca8840bbe9a35))

## [2.10.6](https://github.com/haexhub/fwbg/compare/v2.10.5...v2.10.6) (2026-07-05)


### Bug Fixes

* **tests:** make StabilitySelector noise test deterministic ([30d47c8](https://github.com/haexhub/fwbg/commit/30d47c86571ea911218c7563d0a00bfff8212e2c))
* **tests:** make StabilitySelector noise test deterministic ([624f743](https://github.com/haexhub/fwbg/commit/624f743296315aaf5de80c6f3516fc0bead3a8a4))

## [2.10.5](https://github.com/haexhub/fwbg/compare/v2.10.4...v2.10.5) (2026-07-05)


### Bug Fixes

* **indicators:** use integer choices for opening_range sessions param ([aff7adf](https://github.com/haexhub/fwbg/commit/aff7adf1aef62221ae1054786a0cb8d15ed4cbf5))
* **indicators:** use integer choices for opening_range sessions param ([5c9308a](https://github.com/haexhub/fwbg/commit/5c9308a38b02ffe77cce0e43c7da3e2aefb230fd))

## [2.10.4](https://github.com/haexhub/fwbg/compare/v2.10.3...v2.10.4) (2026-07-04)


### Bug Fixes

* **indicators:** validate range_scope values in previous_day_levels ([049d88e](https://github.com/haexhub/fwbg/commit/049d88e78f3408bd510c291e3367d10eb61a5cb1))
* **indicators:** validate range_scope values in previous_day_levels ([fa052fc](https://github.com/haexhub/fwbg/commit/fa052fc31d595289bf928c00df074d7a7b9b828f))

## [2.10.3](https://github.com/haexhub/fwbg/compare/v2.10.2...v2.10.3) (2026-07-04)


### Bug Fixes

* **data:** ensure full history coverage on first download ([96b1569](https://github.com/haexhub/fwbg/commit/96b156940bbfdfeb0b2a20d67e3d6db7119c5be3))
* **data:** ensure full history coverage on first download ([375c7c7](https://github.com/haexhub/fwbg/commit/375c7c7912b75df7a375ccb3e5e2ad60e7a30e4e))

## [2.10.2](https://github.com/haexhub/fwbg/compare/v2.10.1...v2.10.2) (2026-07-04)


### Bug Fixes

* **api:** enforce a single backtest slot (stale-status-safe) ([5a95da4](https://github.com/haexhub/fwbg/commit/5a95da44410e9eed02cc187d959c200a1bc50edc))
* **api:** single backtest slot — refresh stale job statuses in the gate ([fa1d1f1](https://github.com/haexhub/fwbg/commit/fa1d1f1ccae08704854f81d0f6bfdaa532b94858))
* **feature-selection:** add seed to stability get_default_params ([cbb5dec](https://github.com/haexhub/fwbg/commit/cbb5dec6f438a6f5d841a51c8c410d18ae9d8be8))
* **feature-selection:** add seed to stability get_default_params (fixes broken develop) ([c9f72e2](https://github.com/haexhub/fwbg/commit/c9f72e2f8b15e117765e3acd7ef4031cfe6e1fd5))
* **feature-selection:** seed stability-selection bootstrap for reproducibility ([8b21c90](https://github.com/haexhub/fwbg/commit/8b21c9074e75a42ece3c9e3fdb34c61fa8ccf58f))
* **feature-selection:** seed stability-selection bootstrap for reproducibility ([5463ec0](https://github.com/haexhub/fwbg/commit/5463ec07370e6c0f471e8840ba0c30db121aab35))
* **simulation:** breakeven-scratch threshold must not double-count costs ([f1e0afd](https://github.com/haexhub/fwbg/commit/f1e0afdf11fa0b121f39dd7fc8561ad592724946))
* **simulation:** reclassify breakeven-stop exits as losses ([89bec74](https://github.com/haexhub/fwbg/commit/89bec746af23befefd376651da0f4559e9bb512a))
* **simulation:** reclassify breakeven-stop exits as losses ([a4a1f33](https://github.com/haexhub/fwbg/commit/a4a1f338fe49155101df356447931466f5850c2d))
* **simulation:** use noise_eps threshold instead of spread+slippage ([1f9ae91](https://github.com/haexhub/fwbg/commit/1f9ae915608883de3cffe29ad87011ea173d8103))
* **simulation:** use noise_eps threshold instead of spread+slippage ([c046296](https://github.com/haexhub/fwbg/commit/c04629639df8035d394716a0f9b94b5384ab8f18))

## [2.10.1](https://github.com/haexhub/fwbg/compare/v2.10.0...v2.10.1) (2026-07-03)


### Bug Fixes

* **api:** redirect run-CLI stdout/stderr to files — unread PIPEs deadlocked long backtests ([834f4a3](https://github.com/haexhub/fwbg/commit/834f4a37b4e3379184cd71a75d64207f04bd09be))
* **api:** unread subprocess PIPEs deadlocked long backtest runs ([7f6b861](https://github.com/haexhub/fwbg/commit/7f6b86177660076f2e9ad2cb1c373ecf691d9bdf))
* **compose:** persist the fwbg workspace across redeploys ([190fe45](https://github.com/haexhub/fwbg/commit/190fe452723bab40df4c63a777b6db3cec03bab7))
* **compose:** persist the fwbg workspace across redeploys ([86e335e](https://github.com/haexhub/fwbg/commit/86e335edc9cf488c3f214e4eafec278ca1aeeb3d))

## [2.10.0](https://github.com/haexhub/fwbg/compare/v2.9.0...v2.10.0) (2026-07-03)


### Features

* **data:** full-history ensure defaults + timeframes endpoint ([5bdfa26](https://github.com/haexhub/fwbg/commit/5bdfa269da5c15c8d1d11afe51585c91bcb7a790))
* **data:** full-history ensure defaults + timeframes endpoint ([c9664c5](https://github.com/haexhub/fwbg/commit/c9664c59ca16fb600761c3d16653ac9a379c0848))

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
