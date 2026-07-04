window.BENCHMARK_DATA = {
  "lastUpdate": 1783148622217,
  "repoUrl": "https://github.com/MolCrafts/molpy",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7fa1b1a333d368d761547132af4bf67417f4acf2",
          "message": "Merge pull request #35 from Roy-Kid/dev\n\nchore(release): molpy 0.5.0",
          "timestamp": "2026-06-21T13:54:49+02:00",
          "tree_id": "0c6f346e45e4c72a04a1889c82c2b6217fe1a685",
          "url": "https://github.com/MolCrafts/molpy/commit/7fa1b1a333d368d761547132af4bf67417f4acf2"
        },
        "date": 1782043509114,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 14125.332030682193,
            "unit": "iter/sec",
            "range": "stddev: 0.000009140194603872883",
            "extra": "mean: 70.79479603225329 usec\nrounds: 1613"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 15795.454685438599,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033431982059597546",
            "extra": "mean: 63.30935195691915 usec\nrounds: 10399"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1599.9648955237533,
            "unit": "iter/sec",
            "range": "stddev: 0.000011449258531703663",
            "extra": "mean: 625.0137129869009 usec\nrounds: 1540"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 156.4756068049338,
            "unit": "iter/sec",
            "range": "stddev: 0.0002989298855934115",
            "extra": "mean: 6.390772468750505 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 15906.577384502665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071802384311566186",
            "extra": "mean: 62.867075413361526 usec\nrounds: 14878"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1581.1057063739322,
            "unit": "iter/sec",
            "range": "stddev: 0.0000852554840062686",
            "extra": "mean: 632.4687817953517 usec\nrounds: 1604"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 161.51179774300542,
            "unit": "iter/sec",
            "range": "stddev: 0.00004096283599088842",
            "extra": "mean: 6.191498169014139 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 4763.302759070893,
            "unit": "iter/sec",
            "range": "stddev: 0.000013251743278348423",
            "extra": "mean: 209.93836641932353 usec\nrounds: 3234"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 636.011210899167,
            "unit": "iter/sec",
            "range": "stddev: 0.00002341364369716529",
            "extra": "mean: 1.5722993287905103 msec\nrounds: 587"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 64.06788465464768,
            "unit": "iter/sec",
            "range": "stddev: 0.0008091828443549018",
            "extra": "mean: 15.60844415872964 msec\nrounds: 63"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 74330.82282116033,
            "unit": "iter/sec",
            "range": "stddev: 0.000002572287923846006",
            "extra": "mean: 13.453369168346166 usec\nrounds: 8874"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 64315.95006065362,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030956279978153665",
            "extra": "mean: 15.5482426840767 usec\nrounds: 23442"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 26321.07155347829,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037791568697537372",
            "extra": "mean: 37.992374207419054 usec\nrounds: 8832"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 326401.50054763485,
            "unit": "iter/sec",
            "range": "stddev: 7.553077493983962e-7",
            "extra": "mean: 3.0637114054996832 usec\nrounds: 34124"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 328036.3911470775,
            "unit": "iter/sec",
            "range": "stddev: 7.741294606859576e-7",
            "extra": "mean: 3.0484422673447926 usec\nrounds: 92937"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 329983.0384410641,
            "unit": "iter/sec",
            "range": "stddev: 7.101573578324681e-7",
            "extra": "mean: 3.0304587918345467 usec\nrounds: 88332"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 40.511408331023915,
            "unit": "iter/sec",
            "range": "stddev: 0.0054810294915871765",
            "extra": "mean: 24.68440474418642 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 3.984745727107178,
            "unit": "iter/sec",
            "range": "stddev: 0.002474213560109309",
            "extra": "mean: 250.9570417999981 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7331765250769496,
            "unit": "iter/sec",
            "range": "stddev: 0.03305632665465463",
            "extra": "mean: 1.363928011599998 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 363.3546614931862,
            "unit": "iter/sec",
            "range": "stddev: 0.0026777008462808226",
            "extra": "mean: 2.752132024096112 msec\nrounds: 415"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 31.198895088634803,
            "unit": "iter/sec",
            "range": "stddev: 0.00937434993606822",
            "extra": "mean: 32.05241714999971 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.508987177523841,
            "unit": "iter/sec",
            "range": "stddev: 0.012289858919012266",
            "extra": "mean: 181.5215697142857 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9e6081663ad0d302f4cf614776405dca6cae9a1e",
          "message": "Merge pull request #36 from MolCrafts/dev\n\nchore(release): molpy 0.5.1",
          "timestamp": "2026-07-01T17:21:22+08:00",
          "tree_id": "cdb397714c96fa4febd6a55c2b8bf3c345f8eb8f",
          "url": "https://github.com/MolCrafts/molpy/commit/9e6081663ad0d302f4cf614776405dca6cae9a1e"
        },
        "date": 1782897740778,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 16218.991634563965,
            "unit": "iter/sec",
            "range": "stddev: 0.000007485567760484612",
            "extra": "mean: 61.65611417351744 usec\nrounds: 1524"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 16976.529797659656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029810633019183284",
            "extra": "mean: 58.90485346056163 usec\nrounds: 10591"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1709.8066468390737,
            "unit": "iter/sec",
            "range": "stddev: 0.000009892181954724389",
            "extra": "mean: 584.8614531056502 usec\nrounds: 1642"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 169.2868611529697,
            "unit": "iter/sec",
            "range": "stddev: 0.00006329770208711233",
            "extra": "mean: 5.907132976471148 msec\nrounds: 170"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 17104.285847048126,
            "unit": "iter/sec",
            "range": "stddev: 0.000003306082165845423",
            "extra": "mean: 58.464878857984054 usec\nrounds: 15552"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1711.5289330818703,
            "unit": "iter/sec",
            "range": "stddev: 0.00001815923677335843",
            "extra": "mean: 584.2729156785838 usec\nrounds: 1518"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 172.0222747641101,
            "unit": "iter/sec",
            "range": "stddev: 0.00003670019615746907",
            "extra": "mean: 5.813200653062374 msec\nrounds: 147"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 5219.283440499827,
            "unit": "iter/sec",
            "range": "stddev: 0.000016773392289998205",
            "extra": "mean: 191.5971821419675 usec\nrounds: 3371"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 652.6767156674407,
            "unit": "iter/sec",
            "range": "stddev: 0.00008636656076482775",
            "extra": "mean: 1.5321520991864699 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 67.90993103757282,
            "unit": "iter/sec",
            "range": "stddev: 0.00006397902440499643",
            "extra": "mean: 14.72538676923005 msec\nrounds: 65"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 74277.82729134634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023845269862052285",
            "extra": "mean: 13.46296783934745 usec\nrounds: 8271"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 61081.12317192248,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568922142874905",
            "extra": "mean: 16.37167013424658 usec\nrounds: 22206"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 30000.068233033082,
            "unit": "iter/sec",
            "range": "stddev: 0.000003545810365101284",
            "extra": "mean: 33.33325751902456 usec\nrounds: 9110"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 331817.6674423225,
            "unit": "iter/sec",
            "range": "stddev: 7.71639808677959e-7",
            "extra": "mean: 3.0137033019009545 usec\nrounds: 19262"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 331874.8969621636,
            "unit": "iter/sec",
            "range": "stddev: 7.446419825057571e-7",
            "extra": "mean: 3.013183609708233 usec\nrounds: 89358"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 331548.2326301037,
            "unit": "iter/sec",
            "range": "stddev: 7.739809165122711e-7",
            "extra": "mean: 3.016152407350226 usec\nrounds: 93985"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 39.29810240899715,
            "unit": "iter/sec",
            "range": "stddev: 0.006500037212165738",
            "extra": "mean: 25.44652129999676 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 3.8823175233874414,
            "unit": "iter/sec",
            "range": "stddev: 0.010737447881435467",
            "extra": "mean: 257.57810740000195 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7181288781296088,
            "unit": "iter/sec",
            "range": "stddev: 0.06434563868927785",
            "extra": "mean: 1.392507710599989 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 396.3811027534655,
            "unit": "iter/sec",
            "range": "stddev: 0.0017266284464410567",
            "extra": "mean: 2.5228246075645115 msec\nrounds: 423"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 31.44815292164672,
            "unit": "iter/sec",
            "range": "stddev: 0.00827979018511335",
            "extra": "mean: 31.798369923076457 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.84646602015496,
            "unit": "iter/sec",
            "range": "stddev: 0.014508693113656228",
            "extra": "mean: 171.04349816669165 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "09bd84cdb4667371af74da0fdfe2cd3e332e215a",
          "message": "Merge pull request #37 from MolCrafts/ci/pytest-xdist\n\nci: parallelize test suite with pytest-xdist (-n auto)",
          "timestamp": "2026-07-01T22:30:47+08:00",
          "tree_id": "fb33a35677904993190f1e8fe095aa461f710066",
          "url": "https://github.com/MolCrafts/molpy/commit/09bd84cdb4667371af74da0fdfe2cd3e332e215a"
        },
        "date": 1782916299721,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 20844.03698806268,
            "unit": "iter/sec",
            "range": "stddev: 0.000003459802719402555",
            "extra": "mean: 47.97535144332632 usec\nrounds: 1767"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 18089.210742122625,
            "unit": "iter/sec",
            "range": "stddev: 0.000002552900468142631",
            "extra": "mean: 55.281571664782206 usec\nrounds: 11491"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1896.4584758686265,
            "unit": "iter/sec",
            "range": "stddev: 0.000011891625174009932",
            "extra": "mean: 527.2986531075901 usec\nrounds: 1770"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 185.89444925967854,
            "unit": "iter/sec",
            "range": "stddev: 0.00003629544653842702",
            "extra": "mean: 5.379396770492518 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 18166.003092775667,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024852387122714463",
            "extra": "mean: 55.047882293804314 usec\nrounds: 15887"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1865.8591806686068,
            "unit": "iter/sec",
            "range": "stddev: 0.000009884446009238721",
            "extra": "mean: 535.9461262460668 usec\nrounds: 1806"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 182.86220959738935,
            "unit": "iter/sec",
            "range": "stddev: 0.0001135008360717369",
            "extra": "mean: 5.468598472050163 msec\nrounds: 161"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 6050.188043112055,
            "unit": "iter/sec",
            "range": "stddev: 0.000004792886479843782",
            "extra": "mean: 165.2841189189926 usec\nrounds: 3885"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 722.4529392458163,
            "unit": "iter/sec",
            "range": "stddev: 0.000018090903338658394",
            "extra": "mean: 1.3841732044774027 msec\nrounds: 670"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 73.60304813524077,
            "unit": "iter/sec",
            "range": "stddev: 0.00006961253835139757",
            "extra": "mean: 13.586393842855061 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 80550.98809929074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013083778586788164",
            "extra": "mean: 12.414496998688028 usec\nrounds: 9829"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 72543.41042094892,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010352989770796686",
            "extra": "mean: 13.784849570722999 usec\nrounds: 20621"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 26227.479773416013,
            "unit": "iter/sec",
            "range": "stddev: 0.000002679558369300938",
            "extra": "mean: 38.127948572992246 usec\nrounds: 7078"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 353021.42749588756,
            "unit": "iter/sec",
            "range": "stddev: 3.5839628453409e-7",
            "extra": "mean: 2.832689242387841 usec\nrounds: 39098"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 353741.63026215695,
            "unit": "iter/sec",
            "range": "stddev: 3.4629090899351504e-7",
            "extra": "mean: 2.826922008752272 usec\nrounds: 108743"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 352512.47711111925,
            "unit": "iter/sec",
            "range": "stddev: 3.7871041647520164e-7",
            "extra": "mean: 2.8367790218238977 usec\nrounds: 88692"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 44.871700604073006,
            "unit": "iter/sec",
            "range": "stddev: 0.004828985021163893",
            "extra": "mean: 22.285761104165285 msec\nrounds: 48"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 4.329733024946742,
            "unit": "iter/sec",
            "range": "stddev: 0.009696357982599386",
            "extra": "mean: 230.96112260000154 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7916267473046825,
            "unit": "iter/sec",
            "range": "stddev: 0.02496302656775112",
            "extra": "mean: 1.2632215919999965 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 423.3344716210705,
            "unit": "iter/sec",
            "range": "stddev: 0.0021426424366470607",
            "extra": "mean: 2.3621983727682507 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 37.371306714867394,
            "unit": "iter/sec",
            "range": "stddev: 0.007225301073194106",
            "extra": "mean: 26.75849703703753 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 6.469077026283638,
            "unit": "iter/sec",
            "range": "stddev: 0.01316304769707417",
            "extra": "mean: 154.58155714285584 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "00978b0736346192ecf68c71f7ce8b6b63a01c14",
          "message": "Merge pull request #39 from MolCrafts/ci/tests-data-skip-con\n\nci: skip the con/ dir via sparse-checkout (keep Windows CI)",
          "timestamp": "2026-07-01T23:58:32+08:00",
          "tree_id": "88d6f4882c96dee05748ec58a60fdebb350f1398",
          "url": "https://github.com/MolCrafts/molpy/commit/00978b0736346192ecf68c71f7ce8b6b63a01c14"
        },
        "date": 1782921569334,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 22623.098587997072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000403749244349656",
            "extra": "mean: 44.20260982863598 usec\nrounds: 1343"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 10330.295010668922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035080040023027398",
            "extra": "mean: 96.80265655213331 usec\nrounds: 7448"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1042.9332050462392,
            "unit": "iter/sec",
            "range": "stddev: 0.000011062245507758095",
            "extra": "mean: 958.8341757281227 usec\nrounds: 1030"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 169.152708866611,
            "unit": "iter/sec",
            "range": "stddev: 0.00003752931497169641",
            "extra": "mean: 5.911817828401268 msec\nrounds: 169"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 17039.616117363894,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027982888186007753",
            "extra": "mean: 58.68676812389977 usec\nrounds: 10704"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1735.9681897418666,
            "unit": "iter/sec",
            "range": "stddev: 0.000009592955838223608",
            "extra": "mean: 576.0474217840921 usec\nrounds: 1726"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 106.7569721871684,
            "unit": "iter/sec",
            "range": "stddev: 0.000056466584764149156",
            "extra": "mean: 9.367069705262722 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 4014.663534574879,
            "unit": "iter/sec",
            "range": "stddev: 0.000007652879924034736",
            "extra": "mean: 249.08687649359686 usec\nrounds: 2761"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 464.6670942388942,
            "unit": "iter/sec",
            "range": "stddev: 0.00007100105282307275",
            "extra": "mean: 2.152078364055366 msec\nrounds: 434"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 49.28237530798252,
            "unit": "iter/sec",
            "range": "stddev: 0.0031898434421741774",
            "extra": "mean: 20.291229749999992 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 75013.29665533188,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015641270699657677",
            "extra": "mean: 13.33096990250622 usec\nrounds: 7376"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 69037.80812061879,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013256862498342657",
            "extra": "mean: 14.484816757983667 usec\nrounds: 21327"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 29030.813779070028,
            "unit": "iter/sec",
            "range": "stddev: 0.000002750491820435582",
            "extra": "mean: 34.44615805847499 usec\nrounds: 11249"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 319035.58625300834,
            "unit": "iter/sec",
            "range": "stddev: 6.961380002754784e-7",
            "extra": "mean: 3.1344465730132027 usec\nrounds: 30640"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 319177.3804620394,
            "unit": "iter/sec",
            "range": "stddev: 5.821415436905976e-7",
            "extra": "mean: 3.1330540984840645 usec\nrounds: 95548"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 321064.9161687735,
            "unit": "iter/sec",
            "range": "stddev: 5.910692152398879e-7",
            "extra": "mean: 3.114634921600503 usec\nrounds: 77167"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 42.76623869578963,
            "unit": "iter/sec",
            "range": "stddev: 0.00569001861947497",
            "extra": "mean: 23.382930800001606 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 4.154129479664805,
            "unit": "iter/sec",
            "range": "stddev: 0.0013382270165379607",
            "extra": "mean: 240.7243213999891 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7207021025571042,
            "unit": "iter/sec",
            "range": "stddev: 0.06788216912617574",
            "extra": "mean: 1.387535843799992 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 364.2725407145938,
            "unit": "iter/sec",
            "range": "stddev: 0.0028885742894276143",
            "extra": "mean: 2.7451973130840415 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 32.51307399474351,
            "unit": "iter/sec",
            "range": "stddev: 0.008728615496843065",
            "extra": "mean: 30.75685800000557 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.664884879904703,
            "unit": "iter/sec",
            "range": "stddev: 0.01482107314216127",
            "extra": "mean: 176.52609385714868 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b90fa48527d0e3f402f29bc2a461a23f5092944b",
          "message": "Merge pull request #40: Release molpy 0.6.0\n\nRelease molpy 0.6.0",
          "timestamp": "2026-07-04T15:02:35+08:00",
          "tree_id": "c5b2d0c9fea568e33539d24099791a206aa6931a",
          "url": "https://github.com/MolCrafts/molpy/commit/b90fa48527d0e3f402f29bc2a461a23f5092944b"
        },
        "date": 1783148621275,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 13168.094891055312,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023366789870459076",
            "extra": "mean: 75.94112954633017 usec\nrounds: 5334"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 114045.55539626908,
            "unit": "iter/sec",
            "range": "stddev: 7.855648495970154e-7",
            "extra": "mean: 8.76842588494873 usec\nrounds: 31640"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48839.21072598396,
            "unit": "iter/sec",
            "range": "stddev: 9.263274025351486e-7",
            "extra": "mean: 20.475351364922226 usec\nrounds: 19891"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 67.31729452197094,
            "unit": "iter/sec",
            "range": "stddev: 0.0012574358133897333",
            "extra": "mean: 14.855023617647339 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 42381.738914376765,
            "unit": "iter/sec",
            "range": "stddev: 0.000003967271772199533",
            "extra": "mean: 23.59506772528343 usec\nrounds: 14736"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 26793.652396872312,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022872625963799146",
            "extra": "mean: 37.322272648305784 usec\nrounds: 10365"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 346.6973079824697,
            "unit": "iter/sec",
            "range": "stddev: 0.00018466133550514352",
            "extra": "mean: 2.884360440579376 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 4095.518675767483,
            "unit": "iter/sec",
            "range": "stddev: 0.000022532203679367836",
            "extra": "mean: 244.1693175315833 usec\nrounds: 2447"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 509667.1401837861,
            "unit": "iter/sec",
            "range": "stddev: 2.696459623265208e-7",
            "extra": "mean: 1.9620648873682531 usec\nrounds: 77072"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 695.6347544286557,
            "unit": "iter/sec",
            "range": "stddev: 0.00003312741861152656",
            "extra": "mean: 1.4375359966327848 msec\nrounds: 594"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 56939.60729596819,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011956042616184934",
            "extra": "mean: 17.562467454369123 usec\nrounds: 14303"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 369.80956734355897,
            "unit": "iter/sec",
            "range": "stddev: 0.000041144009340049924",
            "extra": "mean: 2.7040944537570173 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 216979.07849709113,
            "unit": "iter/sec",
            "range": "stddev: 5.363724926312794e-7",
            "extra": "mean: 4.608739270746816 usec\nrounds: 37188"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 126981.92340458691,
            "unit": "iter/sec",
            "range": "stddev: 5.751458027530185e-7",
            "extra": "mean: 7.875136658733879 usec\nrounds: 34041"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 150651.05230090598,
            "unit": "iter/sec",
            "range": "stddev: 6.057317446442426e-7",
            "extra": "mean: 6.637856056940309 usec\nrounds: 39071"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 78150.05838299789,
            "unit": "iter/sec",
            "range": "stddev: 8.113069213663634e-7",
            "extra": "mean: 12.795895750956436 usec\nrounds: 19722"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 4018.548063947517,
            "unit": "iter/sec",
            "range": "stddev: 0.000007184452382473168",
            "extra": "mean: 248.8460966714619 usec\nrounds: 2824"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 13754.25754575337,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028801159083021518",
            "extra": "mean: 72.70476044770226 usec\nrounds: 9739"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 13113.84531707039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019970521196799455",
            "extra": "mean: 76.25528407737832 usec\nrounds: 10036"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 83.71674368309242,
            "unit": "iter/sec",
            "range": "stddev: 0.000053795709064747024",
            "extra": "mean: 11.945041768292784 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1383.3861584413148,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663979391601152",
            "extra": "mean: 722.8639623853959 usec\nrounds: 1090"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 145.56312775507791,
            "unit": "iter/sec",
            "range": "stddev: 0.000037328793970782775",
            "extra": "mean: 6.869871618055523 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 50706.79797322382,
            "unit": "iter/sec",
            "range": "stddev: 9.852382286662984e-7",
            "extra": "mean: 19.721221610720896 usec\nrounds: 25441"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1003.4017754122075,
            "unit": "iter/sec",
            "range": "stddev: 0.00009927453198994107",
            "extra": "mean: 996.6097574316032 usec\nrounds: 841"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1621.3565236065065,
            "unit": "iter/sec",
            "range": "stddev: 0.000009800673041399601",
            "extra": "mean: 616.7674940337145 usec\nrounds: 1257"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 113462.32300347443,
            "unit": "iter/sec",
            "range": "stddev: 7.613569920344988e-7",
            "extra": "mean: 8.813498380157244 usec\nrounds: 50006"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 95878.01523258195,
            "unit": "iter/sec",
            "range": "stddev: 6.410956496709551e-7",
            "extra": "mean: 10.429919701343305 usec\nrounds: 47672"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 82515.78595509518,
            "unit": "iter/sec",
            "range": "stddev: 6.455769719327548e-7",
            "extra": "mean: 12.11889323267425 usec\nrounds: 44892"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 109374.18642373853,
            "unit": "iter/sec",
            "range": "stddev: 6.151272666375467e-7",
            "extra": "mean: 9.14292515169704 usec\nrounds: 57169"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 44374.57171131643,
            "unit": "iter/sec",
            "range": "stddev: 9.522945488733019e-7",
            "extra": "mean: 22.535428770008377 usec\nrounds: 16552"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 176214.6057577516,
            "unit": "iter/sec",
            "range": "stddev: 5.860860617031865e-7",
            "extra": "mean: 5.674898489258802 usec\nrounds: 29721"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 174490.0409618923,
            "unit": "iter/sec",
            "range": "stddev: 4.368823237698898e-7",
            "extra": "mean: 5.730986103776518 usec\nrounds: 37996"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 176453.78639971622,
            "unit": "iter/sec",
            "range": "stddev: 4.704925405772355e-7",
            "extra": "mean: 5.66720624364912 usec\nrounds: 37606"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 158358.70956749443,
            "unit": "iter/sec",
            "range": "stddev: 5.062342890293766e-7",
            "extra": "mean: 6.314777398295152 usec\nrounds: 36891"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 160822.25680064436,
            "unit": "iter/sec",
            "range": "stddev: 4.965981758498485e-7",
            "extra": "mean: 6.218044814777112 usec\nrounds: 36305"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 170337.95330771658,
            "unit": "iter/sec",
            "range": "stddev: 4.5455690786067215e-7",
            "extra": "mean: 5.870682255959092 usec\nrounds: 38937"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.3910565601384,
            "unit": "iter/sec",
            "range": "stddev: 0.003906625114844373",
            "extra": "mean: 42.75138224000273 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 937.1171691404603,
            "unit": "iter/sec",
            "range": "stddev: 0.000011563851993814128",
            "extra": "mean: 1.0671024210528732 msec\nrounds: 760"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 6307.883230146071,
            "unit": "iter/sec",
            "range": "stddev: 0.000003321076008583584",
            "extra": "mean: 158.53178689499032 usec\nrounds: 5082"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 26492.681476726968,
            "unit": "iter/sec",
            "range": "stddev: 0.000006391320306811253",
            "extra": "mean: 37.746273470976135 usec\nrounds: 7946"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1762.398912154542,
            "unit": "iter/sec",
            "range": "stddev: 0.00001505373967756416",
            "extra": "mean: 567.4084301252177 usec\nrounds: 1195"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1576.0456235611525,
            "unit": "iter/sec",
            "range": "stddev: 0.000028234672550973512",
            "extra": "mean: 634.4993983996801 usec\nrounds: 1250"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3093.5965695270743,
            "unit": "iter/sec",
            "range": "stddev: 0.000009594605810296296",
            "extra": "mean: 323.248354310424 usec\nrounds: 2320"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1932.376101863322,
            "unit": "iter/sec",
            "range": "stddev: 0.000011280581614149692",
            "extra": "mean: 517.4976025814723 usec\nrounds: 1472"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 6799.922943598626,
            "unit": "iter/sec",
            "range": "stddev: 0.000005123948254373689",
            "extra": "mean: 147.0604899929622 usec\nrounds: 4247"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 104.37429733074335,
            "unit": "iter/sec",
            "range": "stddev: 0.0001917957118486497",
            "extra": "mean: 9.580902823529247 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5345.545472694745,
            "unit": "iter/sec",
            "range": "stddev: 0.000006511232420397511",
            "extra": "mean: 187.0716478062041 usec\nrounds: 3305"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 13064.901394325745,
            "unit": "iter/sec",
            "range": "stddev: 0.000002651004540155687",
            "extra": "mean: 76.54095272654051 usec\nrounds: 10471"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 20587.434618092204,
            "unit": "iter/sec",
            "range": "stddev: 0.000005673122545061686",
            "extra": "mean: 48.5733175866993 usec\nrounds: 3895"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 16726.023027448937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024615810478135114",
            "extra": "mean: 59.7870754069218 usec\nrounds: 11365"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 16612.487415310166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024127607949478814",
            "extra": "mean: 60.195681417244835 usec\nrounds: 14169"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5746.854503291672,
            "unit": "iter/sec",
            "range": "stddev: 0.000004600633327086894",
            "extra": "mean: 174.00823344791868 usec\nrounds: 4078"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 80021.92351092472,
            "unit": "iter/sec",
            "range": "stddev: 8.696337945118786e-7",
            "extra": "mean: 12.496575389911472 usec\nrounds: 15002"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 346600.0200191098,
            "unit": "iter/sec",
            "range": "stddev: 3.617596965337997e-7",
            "extra": "mean: 2.8851700584000683 usec\nrounds: 84912"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 43.945599086019776,
            "unit": "iter/sec",
            "range": "stddev: 0.007066658445738192",
            "extra": "mean: 22.755407157895036 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 380.90637461030855,
            "unit": "iter/sec",
            "range": "stddev: 0.003319040941815213",
            "extra": "mean: 2.6253170507399974 msec\nrounds: 473"
          }
        ]
      }
    ]
  }
}