# Changelog

## [2.21.0](https://github.com/haexhub/fwbg/compare/v2.20.4...v2.21.0) (2026-10-23)


### Features

* **api:** add research register endpoints ([a1491e7](https://github.com/haexhub/fwbg/commit/a1491e793ad8b58ffcd8e2cabf30f8593f6aad2d))
* **api:** pass research fields to runs and preflight the gate ([a293614](https://github.com/haexhub/fwbg/commit/a293614da6769f33cbe921d49e797cec29e151c1))
* **api:** start holdout looks ([e1e2630](https://github.com/haexhub/fwbg/commit/e1e26306ee582e4cd5921366662e4622f76bad0c))
* **cli:** gate every backtest run through the research register ([7904e84](https://github.com/haexhub/fwbg/commit/7904e8438f4cfe5c95e15c04bbfd730abecaef80))
* **config:** resolve preset references in one place ([188dbfa](https://github.com/haexhub/fwbg/commit/188dbfaf4cf654d3f3c352dd6ae939ee053ee874))
* **exits:** allow column patterns for ORB stop and range columns ([ac31350](https://github.com/haexhub/fwbg/commit/ac31350ae81ee9773279665de5248dffb01159df))
* **optimizer:** evaluate a holdout look once after the cutoff ([ffebab9](https://github.com/haexhub/fwbg/commit/ffebab9e2dd95907aed9eb95a2b269adad225c88))
* **optimizer:** store per-trade return statistics; no Sharpe without an edge ([67391a5](https://github.com/haexhub/fwbg/commit/67391a59e60d73493a63034fe644e45e15b5b122))
* **research:** add a machine-readable --check-json ([03a9244](https://github.com/haexhub/fwbg/commit/03a9244644b8fcc83880cc739241270fae51c274))
* **research:** add append-only trial ledger ([817ca50](https://github.com/haexhub/fwbg/commit/817ca506b59f234c3ebefa819bf924820d08e073))
* **research:** add card editing helpers and refuse unknown criteria ([66d325a](https://github.com/haexhub/fwbg/commit/66d325a6126c6bac7ac7af6afc0c83b5eb80707d))
* **research:** add config fingerprints ([a6e8b2b](https://github.com/haexhub/fwbg/commit/a6e8b2bac442cb29b10a6621628b57b66993282d))
* **research:** add list and show views with coverage ([d01b49f](https://github.com/haexhub/fwbg/commit/d01b49f6f58d88b7e8710dda6fd4ccd152772143))
* **research:** add new, amend, verdict and reopen commands ([9a1cea9](https://github.com/haexhub/fwbg/commit/9a1cea96e9654bb326e744a296f3b84bf48a106e))
* **research:** add pre-run gate ([8bd9328](https://github.com/haexhub/fwbg/commit/8bd9328cb090f7928619d7fa8c67c29e3f0f67f1))
* **research:** add reconcile ([ee5afe9](https://github.com/haexhub/fwbg/commit/ee5afe9346aca1288d45e92628714860f999bc87))
* **research:** add register_trial for study scripts ([5c0ef45](https://github.com/haexhub/fwbg/commit/5c0ef45c73d153fa4e0d01c9f79f454ef3e2519a))
* **research:** add research directory and repository setup ([14db04e](https://github.com/haexhub/fwbg/commit/14db04e1d2f8f65cd7d28c3f9be083816704b645))
* **research:** add research register core ([c5cfdfd](https://github.com/haexhub/fwbg/commit/c5cfdfdc4b92358c6db31697d9c037a65242c7de))
* **research:** add the deflated Sharpe ratio ([e53d34c](https://github.com/haexhub/fwbg/commit/e53d34c447a5274db86d20e562df289145ee0904))
* **research:** backfill legacy runs into the research register ([#181](https://github.com/haexhub/fwbg/issues/181)) ([e2d98fe](https://github.com/haexhub/fwbg/commit/e2d98fe2ead13eff45a8036ec2e536ecf2a3100e))
* **research:** gate and record backtest runs ([dab5492](https://github.com/haexhub/fwbg/commit/dab5492d0c213e300ed574abd8e9b432d7fedddc))
* **research:** gate API runs and expose the research register over HTTP ([caa1e34](https://github.com/haexhub/fwbg/commit/caa1e3480a7d2a8cf87128d7ede0de1777d85b27))
* **research:** gate every backtest run through the research register ([#182](https://github.com/haexhub/fwbg/issues/182)) ([b07d79d](https://github.com/haexhub/fwbg/commit/b07d79dbe8a18bd2651eb7409e9cbb4a2e30fd32))
* **research:** gate holdout looks and hand them the cutoff ([7ff7835](https://github.com/haexhub/fwbg/commit/7ff7835400b4cda1b736270908416ae2b69cca79))
* **research:** holdout looks and the deflated Sharpe ratio in verdicts ([15a0c9b](https://github.com/haexhub/fwbg/commit/15a0c9bd8e7207feeaea048480b72f97a30d0c3b))
* **research:** parse and load research cards ([8f4095f](https://github.com/haexhub/fwbg/commit/8f4095f8ac5e63e85a51f0d60e54c3ba09757b22))
* **research:** record gated runs from their start with pending markers ([1a88e37](https://github.com/haexhub/fwbg/commit/1a88e37c0e5f658ed800dd1141022a79a673e5ac))
* **research:** record per-trade return statistics in the ledger ([24aae4e](https://github.com/haexhub/fwbg/commit/24aae4e886d1758b7ae423fb51d36e85f7f0d149))
* **research:** record the actor of verdicts and reopens ([3f22b63](https://github.com/haexhub/fwbg/commit/3f22b63f90a37f0bf0bd900d2defeeeac92efba3))
* **research:** report the deflated Sharpe ratio and holdout results in verdicts ([46f4096](https://github.com/haexhub/fwbg/commit/46f409666c738ea23d474d6e4651948da15461f3))
* **research:** write holdout looks apart from their development ([ad1358c](https://github.com/haexhub/fwbg/commit/ad1358ccd07abf8b075aeb380fd12ec596ddc0fe))
* **results:** store code.patch for runs with uncommitted changes ([dcd095d](https://github.com/haexhub/fwbg/commit/dcd095dd245b312487a8b4398cbfa277c3aa17c2))


### Bug Fixes

* **api:** harden the research preflight ([4f7520a](https://github.com/haexhub/fwbg/commit/4f7520adff3ff4967cfc10b79c38fc5eb5bc40a0))
* **api:** record interrupted container runs at startup ([e4ab3e3](https://github.com/haexhub/fwbg/commit/e4ab3e3105a9ab1c001351642f71d54089066423))
* **api:** refuse blank research card fields ([988fd03](https://github.com/haexhub/fwbg/commit/988fd032a9fd833a91300cd2f64f275ecf5e2ebc))
* **api:** tighten research endpoint errors ([e20daad](https://github.com/haexhub/fwbg/commit/e20daad37ff078617ae1b570c47c5e6190316f33))
* **ci:** run secret-bearing automerge from trusted base ([0d7bed2](https://github.com/haexhub/fwbg/commit/0d7bed2b4f87033947e325b6da5d74efe648a5bc))
* **ci:** use RELEASE_PAT for main auto-merge, not GITHUB_TOKEN ([94fea9d](https://github.com/haexhub/fwbg/commit/94fea9d92881acc4da7c6d572002edfd8d51f228))
* **ci:** use RELEASE_PAT for main auto-merge, not GITHUB_TOKEN ([318df2d](https://github.com/haexhub/fwbg/commit/318df2d51f0b4058b5f20f7c2c52a2f73b9ef80c))
* **cli:** polish the gated run path ([3927c10](https://github.com/haexhub/fwbg/commit/3927c1046f025874c4b1818f775c9e20643e0059))
* **cli:** report holdout looks consistently ([89ad027](https://github.com/haexhub/fwbg/commit/89ad027894c7c3fabbf905ea266b3d6b5707a20f))
* **config:** type the loaded preset as a dict ([c37d47f](https://github.com/haexhub/fwbg/commit/c37d47f99f597bab90a37430c1d6ac56ce2ea492))
* **exits:** deterministic, unambiguous ORB stop columns ([4e444bb](https://github.com/haexhub/fwbg/commit/4e444bbd64687bae87c81c71d783fc91720478df))
* **exits:** list the ORB column params in the defaults ([1ffe846](https://github.com/haexhub/fwbg/commit/1ffe84648d1f39730a413ac6116909324f8dec36))
* **exits:** refuse ambiguous ORB stop and range columns ([ebf5c5c](https://github.com/haexhub/fwbg/commit/ebf5c5c1a618ad51c6a42adc91879b443ec57a38))
* **exits:** scale an explicit ORB stop column by sl_mult ([4d4f7ff](https://github.com/haexhub/fwbg/commit/4d4f7ff41fdfc2f20a62f8034f98a4f785d09b73))
* **exits:** scale the distance to an sl_level by sl_mult ([d388328](https://github.com/haexhub/fwbg/commit/d388328aa467bf6e8005ed125dff594faeb6adae))
* **optimization:** replay each test fold with the known setting ([7aab55f](https://github.com/haexhub/fwbg/commit/7aab55fa1493edad1c4067ac2ae16f359ed6835a))
* **optimization:** select signal settings and indicator variants on inner windows ([f51aaea](https://github.com/haexhub/fwbg/commit/f51aaea66f7495668807ac43a6ee83c5d92525aa))
* **optimizer:** carry the stop column into feature-selection targets ([f6c42f0](https://github.com/haexhub/fwbg/commit/f6c42f0defb277ddcf1e3f1aadfe80bc29c0ff5c))
* **optimizer:** keep named exit columns without forcing them into the model ([776c96c](https://github.com/haexhub/fwbg/commit/776c96c5fac0ceba99ce3df3d4189e1b658e8d52))
* **optimizer:** never run an ungated holdout look ([a36f2a6](https://github.com/haexhub/fwbg/commit/a36f2a65249b450d727a3985a2e02f42610a1751))
* **optimizer:** never zero-fill exit columns in signal folds ([36f4920](https://github.com/haexhub/fwbg/commit/36f4920ed70084acae1d62784625498b0bde2017))
* **optimizer:** record what a holdout look touched ([1afb9f3](https://github.com/haexhub/fwbg/commit/1afb9f3ed932579cdb49f252dc02a9f7f2f765db))
* **optimizer:** reject empty holdout training windows ([59cfa91](https://github.com/haexhub/fwbg/commit/59cfa9153a6f1723526aed7e4995025434aa64da))
* **plugins:** make opening_range feature order independent of the hash seed ([4607cf4](https://github.com/haexhub/fwbg/commit/4607cf4d75408a7fcb752311650a53614b79c372))
* **research:** apply the run-id rules to register_trial ([baeedf0](https://github.com/haexhub/fwbg/commit/baeedf0cdb429e76885b5d84f9642f8e0b4aaa1b))
* **research:** clearer amend refusal ([0df3382](https://github.com/haexhub/fwbg/commit/0df33825e0816ab52bd8a4582086917236a83fe1))
* **research:** close criteria and verdict loopholes ([84483ef](https://github.com/haexhub/fwbg/commit/84483efaab144b9fcf2b38ba5689df196100360b))
* **research:** close review findings ([ac58850](https://github.com/haexhub/fwbg/commit/ac5885099c3eb9da334db1d588da17afa47b3ce8))
* **research:** count a holdout symbol only when its holdout was looked at ([807d031](https://github.com/haexhub/fwbg/commit/807d031e4960c8c4834b42633e99c51761d5f5d1))
* **research:** count pending holdout looks ([21806f8](https://github.com/haexhub/fwbg/commit/21806f8c59b32f01ce84347073f03c527ec60bf2))
* **research:** fingerprint resolved configs, independent of file location ([f3a060d](https://github.com/haexhub/fwbg/commit/f3a060d8e75e153c2ec9d88b94830c0b911e2974))
* **research:** floor the DSR's SR variance and label holdout criteria ([1e89294](https://github.com/haexhub/fwbg/commit/1e89294ae01c6ffc172fd6a52e5c3e98d7d84e6c))
* **research:** freeze the holdout cutoff and close gate loopholes ([7626446](https://github.com/haexhub/fwbg/commit/7626446037ca3b1abfedd972bd5e0b22d7e2cc2b))
* **research:** guard the startup register check and name study duplicates ([8c123c9](https://github.com/haexhub/fwbg/commit/8c123c990647b4501039ad27c4932efd2265dd0b))
* **research:** harden ledger ids, nesting and write guard ([2eef5b6](https://github.com/haexhub/fwbg/commit/2eef5b6839da044bcc92496447c2b4b39df1da21))
* **research:** harden pending markers ([6bb52fe](https://github.com/haexhub/fwbg/commit/6bb52fefb4338952bcfb00470aaa06f23ae945b0))
* **research:** harden register ownership for root writers ([0c7b7e8](https://github.com/haexhub/fwbg/commit/0c7b7e808d82e16696d2cedec477bdf4e7e3d17f))
* **research:** harden register_trial ([1e0d2a6](https://github.com/haexhub/fwbg/commit/1e0d2a65e6dceb6eaa163f2d165eb5e9d444ba8c))
* **research:** ignore headings in code fences when editing cards ([fb58c9b](https://github.com/haexhub/fwbg/commit/fb58c9bf74c69654a826df700bb565eb955ee3c2))
* **research:** keep gated runs out of the legacy backfill ([93e14fc](https://github.com/haexhub/fwbg/commit/93e14fc4b3c152af774c51c0d2512d0eba010f4b))
* **research:** keep reconcile's report from hiding its repair ([9df1cd4](https://github.com/haexhub/fwbg/commit/9df1cd4eb04d5effc71c034c64cb48dc7877dbb8))
* **research:** keep test imports lazy and harden register init ([66f8f7c](https://github.com/haexhub/fwbg/commit/66f8f7c7ac02fca3a6c96e6c6703dea4ee433854))
* **research:** keep the trial ledger readable and validate every record ([099424b](https://github.com/haexhub/fwbg/commit/099424b58a5eacf2839d7753320abddeecea9648))
* **research:** key coverage by item names ([d5776fb](https://github.com/haexhub/fwbg/commit/d5776fbcde94159fbfc077397c3fd0eaaddc5a67))
* **research:** let a root container write a host-owned register ([21a02fc](https://github.com/haexhub/fwbg/commit/21a02fcdffde2097ab32e9ea8ea5f9adb8e6f951))
* **research:** make register init recoverable and isolated ([87c5a1d](https://github.com/haexhub/fwbg/commit/87c5a1dc226dddba266cd7e6face638050b8c5e4))
* **research:** never let one broken run block recording ([f9077bb](https://github.com/haexhub/fwbg/commit/f9077bb3ab90673e734deb7b2ffc3aef9c2da2e9))
* **research:** never record a refused run ([05e9e44](https://github.com/haexhub/fwbg/commit/05e9e448292521e5ec598827d7689a2792a6a5d9))
* **research:** refuse a run id held by a stale marker ([0250135](https://github.com/haexhub/fwbg/commit/025013530b50799d611862e38f75019659321f0b))
* **research:** refuse runs without resolved symbols ([7278e02](https://github.com/haexhub/fwbg/commit/7278e02ac5c18eeb93bffbcbf1d8d0e83ec92174))
* **research:** refuse unknown keys in criteria amendments ([cf588e4](https://github.com/haexhub/fwbg/commit/cf588e467e2a25c6a9b7ae74ff21ed892a9121e7))
* **research:** refuse unresolved preset references in fingerprints ([f85c05f](https://github.com/haexhub/fwbg/commit/f85c05f2f1d17c1b23855fdce0146e20e92f4725))
* **research:** report contaminated holdouts once and pin cutoff ordering ([eb360c6](https://github.com/haexhub/fwbg/commit/eb360c67b4f14e163ab52150895ff3e426c7219e))
* **research:** report non-UTF-8 research cards as card errors ([c9ed093](https://github.com/haexhub/fwbg/commit/c9ed093b89444a0fd63ed86c146583b06caebce9))
* **research:** store the absolute run dir in pending markers ([8762098](https://github.com/haexhub/fwbg/commit/8762098ab1f526b987d69e252bb1662ea878b4da))
* **research:** tie --check-json payloads to the exit code ([98a5105](https://github.com/haexhub/fwbg/commit/98a5105fe84464df68725e0e1f4432af2b29eeaa))
* **research:** validate register_trial's timeframe, symbols and metrics ([6b2ba22](https://github.com/haexhub/fwbg/commit/6b2ba223a698da00af0cdc91d3464fc87445a7da))
* **research:** validate research cards strictly at the parse boundary ([20456c6](https://github.com/haexhub/fwbg/commit/20456c603dfb195b9c89f062d92a13f39b80ffb1))
* **results:** align git metadata with code patch capture ([ae7a252](https://github.com/haexhub/fwbg/commit/ae7a252dd74acbdbe7419d577e2bd3d06b57da2b))
* **results:** capture code state from the fwbg source tree ([495f2ac](https://github.com/haexhub/fwbg/commit/495f2ac7fbc54c5e60618df7850e8b6008322159))
* **results:** keep non-UTF-8 changes in code.patch instead of failing the run ([9526ed9](https://github.com/haexhub/fwbg/commit/9526ed9add45a53bd560b61fc6491feec8386215))
* **results:** make code.patch immune to git config and odd file names ([e2ca765](https://github.com/haexhub/fwbg/commit/e2ca7653d873af49486c61e9d7e0cf792cbfb6bd))
* **results:** never fail a run because code.patch cannot be written ([cc065a2](https://github.com/haexhub/fwbg/commit/cc065a222c075433193692c863f5af5f70297b04))
* **results:** remove a partial code.patch after a failed write ([c0727f8](https://github.com/haexhub/fwbg/commit/c0727f86a792d4b12afc1237f5b433d6bbb44588))
* **review:** harden research API validation ([2c05646](https://github.com/haexhub/fwbg/commit/2c05646a9103bea4d369ddec26821a47b882392d))
* **review:** preserve exit target configuration and API compatibility ([1c12375](https://github.com/haexhub/fwbg/commit/1c12375cd88657317ad27a9982db691701b1d2ae))


### Documentation

* add implementation plan for research register core ([1332b03](https://github.com/haexhub/fwbg/commit/1332b03b20cf998a0f4be464500c770c9da9f778))
* add research register design ([b3e908b](https://github.com/haexhub/fwbg/commit/b3e908b6d01aad9648ffe4bd15eeb8cd402764fe))
* add research register design ([9d209d5](https://github.com/haexhub/fwbg/commit/9d209d56657f6eb021f1f740d4747a6c369267d5))
* address research register review findings ([246c91a](https://github.com/haexhub/fwbg/commit/246c91abdb726ec0959c622371da9f19084c218c))
* align backfill cutoff rules with the frozen holdout cutoff ([da7ed3e](https://github.com/haexhub/fwbg/commit/da7ed3e097c69025fa1caf88d6126a1c565f085a))
* align research register design with the merged core and add step-3 notes ([4605be7](https://github.com/haexhub/fwbg/commit/4605be7d6ec4eedd2b7e4a6b71f2857593722ef6))
* align step-5 plan and design ([f8200f9](https://github.com/haexhub/fwbg/commit/f8200f948b82ccb789cfb118321f4234d56d857c))
* anchor the holdout cutoff in trials and refuse later cutoffs ([8ce3175](https://github.com/haexhub/fwbg/commit/8ce317582d17fac80e70a561b0fdeda27c158a7d))
* document the research gate for AI sessions ([3a37e7c](https://github.com/haexhub/fwbg/commit/3a37e7c63b28dd8d83df66772ed047469c08aeb7))
* fix amend syntax and recovery wording in the register design ([6203946](https://github.com/haexhub/fwbg/commit/62039465ae76b0e2b9529dda5c60437fdf8d1b8f))
* fix note check typing and test counts in research core plan ([f95105c](https://github.com/haexhub/fwbg/commit/f95105cea826aadcf0d2efe62b2f83850c25f74a))
* make max drawdown a per-strategy criterion with a 0.40 default ([043ceb8](https://github.com/haexhub/fwbg/commit/043ceb88c05a55aa5fa4fef5385c064dcd8615c6))
* mark research register core as implemented ([33c243d](https://github.com/haexhub/fwbg/commit/33c243df6945a748e98d2792a41f095d2dbe1921))
* plan research register step 3 (CLI gate and card commands) ([9b8743f](https://github.com/haexhub/fwbg/commit/9b8743f2e3dc882e7eced0d446f68e1d3fa6cb2f))
* plan step 4 of the research register ([ef34a01](https://github.com/haexhub/fwbg/commit/ef34a0107bada50a40c5f09e609884cdfb55d9bb))
* plan step 5 of the research register ([319a4dd](https://github.com/haexhub/fwbg/commit/319a4ddb4a9e0e5c94ba186056560d95635788d0))
* record code.patch review changes and step-3 notes in plan ([9d41f67](https://github.com/haexhub/fwbg/commit/9d41f67644313d8471c85b865ce70bea61d4b6f9))
* record fingerprint review changes in research register plan and design ([a9c79d4](https://github.com/haexhub/fwbg/commit/a9c79d4e57744970762d6f1164feda41c25b5b93))
* record gate review changes in research register plan and design ([7602bb2](https://github.com/haexhub/fwbg/commit/7602bb2f746673b4756beb76c12323da562e8768))
* record ledger review changes in research core plan ([fb8024f](https://github.com/haexhub/fwbg/commit/fb8024fe5a981d17571f292339820d88cea1fb55))
* record register init review changes and step-4 decisions in plan ([ca7981a](https://github.com/haexhub/fwbg/commit/ca7981a6d8a670d21b03e60654baf1a382d23121))
* record step 4 of the research register ([5920cda](https://github.com/haexhub/fwbg/commit/5920cda2f2e7362efadf91ed08a72885ddeaf81e))
* record step 5 of the research register ([507e2ac](https://github.com/haexhub/fwbg/commit/507e2acf2ead0454cf4decb5db8cc8919297d7db))
* settle fingerprint and criteria-hash details in research register design ([24a6739](https://github.com/haexhub/fwbg/commit/24a673939a7d456d497131bbb13e98ff1466008e))
* tidy research register wording ([909351f](https://github.com/haexhub/fwbg/commit/909351f4cee59fda922ce3abf2eb2fd8b34f78d3))

## [2.20.4](https://github.com/haexhub/fwbg/compare/v2.20.3...v2.20.4) (2026-10-18)


### Documentation

* close out Jev signal provider feasibility check (Stage A fail) ([2b69962](https://github.com/haexhub/fwbg/commit/2b6996219bf0d0c5851e52a64a15830e7c2f5876))
* find and correct for a positional bias in the Jev feasibility check ([1cc4714](https://github.com/haexhub/fwbg/commit/1cc4714653b775553d599a3aef94424776fa9650))

## [2.20.3](https://github.com/haexhub/fwbg/compare/v2.20.2...v2.20.3) (2026-09-24)


### Documentation

* add Jev signal provider design ([6663b53](https://github.com/haexhub/fwbg/commit/6663b533ed7b723949b01be8e2097f95479f0877))

## [2.20.2](https://github.com/haexhub/fwbg/compare/v2.20.1...v2.20.2) (2026-07-30)


### Bug Fixes

* **compose:** wire FWBG_API_KEY into the dashboard service ([2a84bfe](https://github.com/haexhub/fwbg/commit/2a84bfe6f0a3c8ec7dc7039448c12aabcb772345))
* **compose:** wire FWBG_API_KEY into the dashboard service ([f4a2a51](https://github.com/haexhub/fwbg/commit/f4a2a510717bc4aceb1387874dead9e6ba8352a7))

## [2.20.1](https://github.com/haexhub/fwbg/compare/v2.20.0...v2.20.1) (2026-07-29)


### Bug Fixes

* **api:** build the app on first access, not at import time ([c4726e6](https://github.com/haexhub/fwbg/commit/c4726e6c19de45789e3b371c53b844b64c40765f))
* **api:** build the app on first access, not at import time ([b4dc9f9](https://github.com/haexhub/fwbg/commit/b4dc9f9005619cbc825edbd935b84b297e5d18a4))
* **compose:** align the ANTHROPIC_MODEL fallback with the documented default ([19a4a41](https://github.com/haexhub/fwbg/commit/19a4a41c804ee84096859a82947fe9737b1e394e))
* **env:** point ANTHROPIC_MODEL at a model claude-proxy serves ([35aaca1](https://github.com/haexhub/fwbg/commit/35aaca1c73be2ec960788e3db93a323b7b7622fb))
* **env:** point ANTHROPIC_MODEL at a model claude-proxy serves ([975f87e](https://github.com/haexhub/fwbg/commit/975f87eff33d76ad07128692e55a61cb2adf1209))

## [2.20.0](https://github.com/haexhub/fwbg/compare/v2.19.0...v2.20.0) (2026-07-29)


### Features

* **compose:** enable the MCP tool bridge between agents and claude-proxy ([4b7acab](https://github.com/haexhub/fwbg/commit/4b7acab27b2b78c5389aafd582b93beacfae8ae6))
* **compose:** enable the MCP tool bridge between agents and claude-proxy ([d185ced](https://github.com/haexhub/fwbg/commit/d185cedec0d391c9a4f7a73a4525d03decdf30b5))

## [2.19.0](https://github.com/haexhub/fwbg/compare/v2.18.1...v2.19.0) (2026-07-20)


### Features

* **api:** auth fail-closed, strategy path validation, run concurrency default ([#146](https://github.com/haexhub/fwbg/issues/146)) ([711e85c](https://github.com/haexhub/fwbg/commit/711e85c53569fa743b61f6851ec75065a4ef1cb9))


### Bug Fixes

* **api:** constrain agent plugin registration paths ([#145](https://github.com/haexhub/fwbg/issues/145)) ([1c6abbf](https://github.com/haexhub/fwbg/commit/1c6abbf5fa7bb9b9ecf184633d9a7c4cdab716f5))
* default the agents service to claude-sonnet-5 ([a42cd69](https://github.com/haexhub/fwbg/commit/a42cd69f0119e6fa5b2e86885897c919d5bdd0f7))
* **deploy:** share backtest artifacts with agents ([#147](https://github.com/haexhub/fwbg/issues/147)) ([1d8a539](https://github.com/haexhub/fwbg/commit/1d8a539611eb6cc068078fb0956e6daa5537ff19))
* **optimization:** enforce valid half-open gate windows ([#144](https://github.com/haexhub/fwbg/issues/144)) ([ee68574](https://github.com/haexhub/fwbg/commit/ee6857408421c519013120821c4b04e6445833fd))
* **validation:** remove temporal leakage from meta-labeling ([#148](https://github.com/haexhub/fwbg/issues/148)) ([2862cbb](https://github.com/haexhub/fwbg/commit/2862cbb7971580f41a15815ed835fc41c3a8fe71))

## [2.18.1](https://github.com/haexhub/fwbg/compare/v2.18.0...v2.18.1) (2026-07-18)


### Bug Fixes

* **optimization:** fail loudly on signal strategies with no signal source ([8962a80](https://github.com/haexhub/fwbg/commit/8962a80a4be9bca0e68d96fae9e8ad857e320d5c))
* **optimization:** fail loudly when a signal strategy has no signal source ([3049e0e](https://github.com/haexhub/fwbg/commit/3049e0e9835f289ee52d9c8af2bd48c540428494))

## [2.18.0](https://github.com/haexhub/fwbg/compare/v2.17.2...v2.18.0) (2026-07-17)


### Features

* **api:** expose depends_on on GET /api/plugins ([334cdd0](https://github.com/haexhub/fwbg/commit/334cdd095b9d3a370785b779269424f8a83b4e49))
* **api:** expose depends_on on GET /api/plugins ([49962aa](https://github.com/haexhub/fwbg/commit/49962aa13273f1007e4607525e9c0f0001dc374a))


### Bug Fixes

* **broker:** fail-loud IG/yfinance timeframe mapping + collect legacy adapter tests in CI ([8eda269](https://github.com/haexhub/fwbg/commit/8eda2692105a741f7359a77bc7fc7cc8cc4d76fb))
* **broker:** fail-loud IG/yfinance timeframe mapping instead of silent HOUR fallback ([0b4d1ee](https://github.com/haexhub/fwbg/commit/0b4d1eee096b84b675cafb0f69d2c4b1ebcb44e5))
* **review:** normalize TIMEFRAME env to canonical, treat same-canonical MTF spellings as equal ([1a24ad0](https://github.com/haexhub/fwbg/commit/1a24ad0d419f990877df190feef711d3a7984a1a))
* **timeframe:** canonical Timeframe enum + adaptive walk-forward folds ([557a636](https://github.com/haexhub/fwbg/commit/557a636440b7c95541398619767fc68a1715eb22))
* **timeframe:** canonical Timeframe enum + adaptive walk-forward folds ([6a089bb](https://github.com/haexhub/fwbg/commit/6a089bbdf37fc784436701d8219db344f12832c6))

## [2.17.2](https://github.com/haexhub/fwbg/compare/v2.17.1...v2.17.2) (2026-07-16)


### Bug Fixes

* **sdk:** ship py.typed marker ([229d14c](https://github.com/haexhub/fwbg/commit/229d14c9078d39deed5a602fc8ef9f242d1cb3ec))
* **sdk:** ship py.typed so downstream type-checkers see fwbg-sdk types ([#129](https://github.com/haexhub/fwbg/issues/129)) ([b2d763b](https://github.com/haexhub/fwbg/commit/b2d763b52197ffafcc874deb27cfb83560056cd6))

## [2.17.1](https://github.com/haexhub/fwbg/compare/v2.17.0...v2.17.1) (2026-07-16)


### Bug Fixes

* **data:** resolve Dukascopy index majors and fwbg canonical asset names ([#125](https://github.com/haexhub/fwbg/issues/125)) ([#126](https://github.com/haexhub/fwbg/issues/126)) ([8ea51ae](https://github.com/haexhub/fwbg/commit/8ea51aebe0d17974605a7d26f8bee3e1e61beb46))

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
