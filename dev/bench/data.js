window.BENCHMARK_DATA = {
  "lastUpdate": 1785497322645,
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
          "id": "c17ccc6b6e3c6a1872006d53f95bc0a6a67edb21",
          "message": "Merge pull request #41 from MolCrafts/nightly\n\nRelease molpy 0.7.0",
          "timestamp": "2026-07-08T10:57:29+08:00",
          "tree_id": "07ffe820f002f9819fd9fedf39f659b287dc91fa",
          "url": "https://github.com/MolCrafts/molpy/commit/c17ccc6b6e3c6a1872006d53f95bc0a6a67edb21"
        },
        "date": 1783481188933,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7826.655456611844,
            "unit": "iter/sec",
            "range": "stddev: 0.000004932441962875351",
            "extra": "mean: 127.76849645977639 usec\nrounds: 4802"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 93297.4734234125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015431187243908907",
            "extra": "mean: 10.718403867827094 usec\nrounds: 51502"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48149.747721397805,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017095439017463339",
            "extra": "mean: 20.768540798721546 usec\nrounds: 19155"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 63.165236266028224,
            "unit": "iter/sec",
            "range": "stddev: 0.0009359556944399249",
            "extra": "mean: 15.831493066666862 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33566.65024885197,
            "unit": "iter/sec",
            "range": "stddev: 0.000007327278303170327",
            "extra": "mean: 29.791474352857165 usec\nrounds: 10469"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18786.055604125846,
            "unit": "iter/sec",
            "range": "stddev: 0.000004630609012411417",
            "extra": "mean: 53.23097200778951 usec\nrounds: 11789"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.4255336875144,
            "unit": "iter/sec",
            "range": "stddev: 0.00002774153839844806",
            "extra": "mean: 2.736535649025358 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3628.277025263121,
            "unit": "iter/sec",
            "range": "stddev: 0.000014640826280952004",
            "extra": "mean: 275.61291297140696 usec\nrounds: 2413"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 468084.9507792974,
            "unit": "iter/sec",
            "range": "stddev: 5.264288997262367e-7",
            "extra": "mean: 2.13636434654679 usec\nrounds: 69508"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 555.5618424460084,
            "unit": "iter/sec",
            "range": "stddev: 0.00003770850980292169",
            "extra": "mean: 1.7999796307054399 msec\nrounds: 482"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52591.3183629636,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030723914099693073",
            "extra": "mean: 19.014545197334897 usec\nrounds: 12722"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 289.35892436930806,
            "unit": "iter/sec",
            "range": "stddev: 0.00041979319785647763",
            "extra": "mean: 3.4559155283688523 msec\nrounds: 282"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217026.37048664235,
            "unit": "iter/sec",
            "range": "stddev: 8.427483732584331e-7",
            "extra": "mean: 4.6077349851895 usec\nrounds: 54363"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 119964.06114934671,
            "unit": "iter/sec",
            "range": "stddev: 0.000001729108664467642",
            "extra": "mean: 8.33582983452912 usec\nrounds: 36441"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 144341.27412757126,
            "unit": "iter/sec",
            "range": "stddev: 9.826799582281276e-7",
            "extra": "mean: 6.9280253069969655 usec\nrounds: 41451"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75142.33780722767,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013886242831117165",
            "extra": "mean: 13.30807676712733 usec\nrounds: 29101"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3800.677473043657,
            "unit": "iter/sec",
            "range": "stddev: 0.000008508821823275596",
            "extra": "mean: 263.11098668395573 usec\nrounds: 3079"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15178.591422847074,
            "unit": "iter/sec",
            "range": "stddev: 0.000003639080504177093",
            "extra": "mean: 65.88226615644868 usec\nrounds: 11760"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17542.176374756906,
            "unit": "iter/sec",
            "range": "stddev: 0.000009303523465514394",
            "extra": "mean: 57.005469483193345 usec\nrounds: 12616"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.09200212183234,
            "unit": "iter/sec",
            "range": "stddev: 0.00004337737044459171",
            "extra": "mean: 12.643503428571861 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1269.7757242755647,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703895215657085",
            "extra": "mean: 787.5406505904988 usec\nrounds: 1016"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.50842534850125,
            "unit": "iter/sec",
            "range": "stddev: 0.000026978533706112018",
            "extra": "mean: 7.272281661764366 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44123.656926734846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017627399776023422",
            "extra": "mean: 22.66357935065198 usec\nrounds: 26364"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1054.7186515430797,
            "unit": "iter/sec",
            "range": "stddev: 0.00009984027029243929",
            "extra": "mean: 948.1201442081024 usec\nrounds: 846"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1260.89812069185,
            "unit": "iter/sec",
            "range": "stddev: 0.000012416586316025378",
            "extra": "mean: 793.0854869157105 usec\nrounds: 1070"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 108573.36523846388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010592231228972243",
            "extra": "mean: 9.210362023905784 usec\nrounds: 53955"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93111.0826030859,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011895182528229346",
            "extra": "mean: 10.739860090154917 usec\nrounds: 48810"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 79831.69566112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013217102225695486",
            "extra": "mean: 12.526352994491441 usec\nrounds: 49658"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 101836.71913978973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010777156756384749",
            "extra": "mean: 9.81964077836517 usec\nrounds: 57761"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40246.643441849075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019913569183200437",
            "extra": "mean: 24.846792539230357 usec\nrounds: 16379"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 177153.42335372357,
            "unit": "iter/sec",
            "range": "stddev: 9.13886157526948e-7",
            "extra": "mean: 5.644824588025559 usec\nrounds: 31617"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 177179.96185777764,
            "unit": "iter/sec",
            "range": "stddev: 8.849933268135969e-7",
            "extra": "mean: 5.643979090607887 usec\nrounds: 47921"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 175406.20937330104,
            "unit": "iter/sec",
            "range": "stddev: 9.043243831568102e-7",
            "extra": "mean: 5.70105245175096 usec\nrounds: 38054"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 158441.86508619023,
            "unit": "iter/sec",
            "range": "stddev: 9.66594565773378e-7",
            "extra": "mean: 6.311463194756093 usec\nrounds: 42820"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 169120.01945788827,
            "unit": "iter/sec",
            "range": "stddev: 9.227371001672191e-7",
            "extra": "mean: 5.912960530666241 usec\nrounds: 45453"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180328.98759851686,
            "unit": "iter/sec",
            "range": "stddev: 8.820574485094563e-7",
            "extra": "mean: 5.545420141915246 usec\nrounds: 46927"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 22.7253332368514,
            "unit": "iter/sec",
            "range": "stddev: 0.0002983560891486726",
            "extra": "mean: 44.00375517391314 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 858.2031694543494,
            "unit": "iter/sec",
            "range": "stddev: 0.000013097609816807665",
            "extra": "mean: 1.165225246879251 msec\nrounds: 721"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4509.038398898271,
            "unit": "iter/sec",
            "range": "stddev: 0.000006818343430022483",
            "extra": "mean: 221.77677622890457 usec\nrounds: 4232"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34770.30947794308,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047882466575349186",
            "extra": "mean: 28.76016966815785 usec\nrounds: 8829"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1455.494498028088,
            "unit": "iter/sec",
            "range": "stddev: 0.000024125990750188945",
            "extra": "mean: 687.0517211537424 usec\nrounds: 1144"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1297.7732088036473,
            "unit": "iter/sec",
            "range": "stddev: 0.000026462563990059287",
            "extra": "mean: 770.550658016627 usec\nrounds: 1079"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2576.872416955739,
            "unit": "iter/sec",
            "range": "stddev: 0.000019699139608737743",
            "extra": "mean: 388.0673305438141 usec\nrounds: 2151"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1482.7926941294104,
            "unit": "iter/sec",
            "range": "stddev: 0.000028802707175229894",
            "extra": "mean: 674.4031070284766 usec\nrounds: 1252"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5553.625727981303,
            "unit": "iter/sec",
            "range": "stddev: 0.000021837728260150157",
            "extra": "mean: 180.0625481406886 usec\nrounds: 3926"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 110.45937995297861,
            "unit": "iter/sec",
            "range": "stddev: 0.0001265083471276266",
            "extra": "mean: 9.053101696077684 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5749.0962455919425,
            "unit": "iter/sec",
            "range": "stddev: 0.000016107077981171976",
            "extra": "mean: 173.940382502161 usec\nrounds: 3749"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10459.836157176556,
            "unit": "iter/sec",
            "range": "stddev: 0.000004856223835526731",
            "extra": "mean: 95.60379196894915 usec\nrounds: 7720"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 16615.353488151486,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075616329449903075",
            "extra": "mean: 60.185297936219435 usec\nrounds: 4991"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 14423.064850306197,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035231467896110042",
            "extra": "mean: 69.3333913685322 usec\nrounds: 12883"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 15285.973089003397,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032001418581596654",
            "extra": "mean: 65.41945312721974 usec\nrounds: 14102"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 4793.348198523063,
            "unit": "iter/sec",
            "range": "stddev: 0.000010683251553685931",
            "extra": "mean: 208.62244063724023 usec\nrounds: 3327"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 77652.40210394672,
            "unit": "iter/sec",
            "range": "stddev: 0.000002350673897150595",
            "extra": "mean: 12.877901686304364 usec\nrounds: 13996"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 337542.4914478969,
            "unit": "iter/sec",
            "range": "stddev: 6.726065804733492e-7",
            "extra": "mean: 2.9625899711484474 usec\nrounds: 80109"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 40.756115214574656,
            "unit": "iter/sec",
            "range": "stddev: 0.006549958223831778",
            "extra": "mean: 24.53619523684125 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 348.94811694718624,
            "unit": "iter/sec",
            "range": "stddev: 0.003086869500261501",
            "extra": "mean: 2.8657555419660037 msec\nrounds: 417"
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
          "id": "80180100a8561fe2cade60f6aacac24cf0e5b5af",
          "message": "release: v0.8.0 with molrs core cutover\n\nrelease: v0.8.0 with molrs core cutover",
          "timestamp": "2026-07-17T15:04:15+08:00",
          "tree_id": "2c73f71a27cd40a0ebf0ae1657dcb7733816f07d",
          "url": "https://github.com/MolCrafts/molpy/commit/80180100a8561fe2cade60f6aacac24cf0e5b5af"
        },
        "date": 1784271923368,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7798.7356232275915,
            "unit": "iter/sec",
            "range": "stddev: 0.000005070987827201252",
            "extra": "mean: 128.22591357265924 usec\nrounds: 4929"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113037.50901509088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011946229035938556",
            "extra": "mean: 8.846620990794275 usec\nrounds: 24163"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 49112.863544770626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017348993709298827",
            "extra": "mean: 20.361264398448554 usec\nrounds: 18752"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.933761992225214,
            "unit": "iter/sec",
            "range": "stddev: 0.0008781950050310651",
            "extra": "mean: 16.411263104477186 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33118.846620446384,
            "unit": "iter/sec",
            "range": "stddev: 0.000007582559081339594",
            "extra": "mean: 30.194288208775845 usec\nrounds: 10194"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18504.95495761292,
            "unit": "iter/sec",
            "range": "stddev: 0.000004042478026697646",
            "extra": "mean: 54.03958033351499 usec\nrounds: 12112"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.2448122588015,
            "unit": "iter/sec",
            "range": "stddev: 0.00002113276327773237",
            "extra": "mean: 2.737889674094618 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3778.906662093879,
            "unit": "iter/sec",
            "range": "stddev: 0.000014656094305158643",
            "extra": "mean: 264.62680595712396 usec\nrounds: 2283"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 481341.92526582704,
            "unit": "iter/sec",
            "range": "stddev: 5.270435664983538e-7",
            "extra": "mean: 2.07752524247236 usec\nrounds: 71348"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 835.7438147873004,
            "unit": "iter/sec",
            "range": "stddev: 0.000029220374485205695",
            "extra": "mean: 1.1965389181546062 msec\nrounds: 672"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52466.87144948761,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026206354426091048",
            "extra": "mean: 19.059646065665422 usec\nrounds: 12734"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 453.1001997240627,
            "unit": "iter/sec",
            "range": "stddev: 0.00004024826547449281",
            "extra": "mean: 2.2070173454105704 msec\nrounds: 414"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217271.26301480265,
            "unit": "iter/sec",
            "range": "stddev: 7.85933083203881e-7",
            "extra": "mean: 4.602541477985841 usec\nrounds: 49906"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 122848.78520961516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011830129131300955",
            "extra": "mean: 8.140088632490048 usec\nrounds: 31027"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 151396.0262375101,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010597290442391664",
            "extra": "mean: 6.605193180111609 usec\nrounds: 46599"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 76066.5619522305,
            "unit": "iter/sec",
            "range": "stddev: 0.000001301573178920835",
            "extra": "mean: 13.14638093710606 usec\nrounds: 31202"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3565.4946848792274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000217026327642238",
            "extra": "mean: 280.46599094393895 usec\nrounds: 2871"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15272.790713112094,
            "unit": "iter/sec",
            "range": "stddev: 0.000003824595973712281",
            "extra": "mean: 65.47591850004686 usec\nrounds: 12000"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16966.85020004983,
            "unit": "iter/sec",
            "range": "stddev: 0.000004034712845544282",
            "extra": "mean: 58.93845871268805 usec\nrounds: 12413"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 73.98116116261315,
            "unit": "iter/sec",
            "range": "stddev: 0.0018223132205452137",
            "extra": "mean: 13.516954644736725 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1269.3973341628496,
            "unit": "iter/sec",
            "range": "stddev: 0.000008692141599699974",
            "extra": "mean: 787.7754057672466 usec\nrounds: 971"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.8139860546702,
            "unit": "iter/sec",
            "range": "stddev: 0.00022767983596380323",
            "extra": "mean: 7.256157583333408 msec\nrounds: 132"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44500.931337291964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018015727031199545",
            "extra": "mean: 22.471439809215767 usec\nrounds: 24954"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1090.5099679225864,
            "unit": "iter/sec",
            "range": "stddev: 0.00008781326145809483",
            "extra": "mean: 917.0021635886491 usec\nrounds: 758"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1247.0308323963427,
            "unit": "iter/sec",
            "range": "stddev: 0.000012654833474283649",
            "extra": "mean: 801.9047917831841 usec\nrounds: 1071"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 105977.29252295688,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012890272065892125",
            "extra": "mean: 9.435983654549199 usec\nrounds: 54633"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 91731.31327234517,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015358090468025965",
            "extra": "mean: 10.901402850638968 usec\nrounds: 40623"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 78319.41455680791,
            "unit": "iter/sec",
            "range": "stddev: 0.000001272817019432317",
            "extra": "mean: 12.768226188343936 usec\nrounds: 50259"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 102613.71619037306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011184629989974478",
            "extra": "mean: 9.745285885024963 usec\nrounds: 58576"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40205.79806194747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022782048594785728",
            "extra": "mean: 24.872034587131946 usec\nrounds: 16191"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 177996.9183961199,
            "unit": "iter/sec",
            "range": "stddev: 9.484761663954401e-7",
            "extra": "mean: 5.618074790343105 usec\nrounds: 16339"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 179889.6237889417,
            "unit": "iter/sec",
            "range": "stddev: 9.087099295208699e-7",
            "extra": "mean: 5.5589643189941045 usec\nrounds: 48289"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 178150.71391634218,
            "unit": "iter/sec",
            "range": "stddev: 9.99876296814382e-7",
            "extra": "mean: 5.613224769167021 usec\nrounds: 45807"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 161910.33957373368,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010538223304843846",
            "extra": "mean: 6.176257814249112 usec\nrounds: 40087"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 170423.61297688424,
            "unit": "iter/sec",
            "range": "stddev: 9.421281458455298e-7",
            "extra": "mean: 5.867731487042451 usec\nrounds: 41417"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 181019.4401579704,
            "unit": "iter/sec",
            "range": "stddev: 8.883919521596513e-7",
            "extra": "mean: 5.52426854887701 usec\nrounds: 35420"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.40335087715104,
            "unit": "iter/sec",
            "range": "stddev: 0.00020935997989974617",
            "extra": "mean: 42.728923958334164 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 850.6477836806818,
            "unit": "iter/sec",
            "range": "stddev: 0.000022150933629213978",
            "extra": "mean: 1.1755746845928214 msec\nrounds: 688"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 5367.536202754314,
            "unit": "iter/sec",
            "range": "stddev: 0.000011097866597189878",
            "extra": "mean: 186.30521755714602 usec\nrounds: 4192"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 35342.45510828473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038260718447790895",
            "extra": "mean: 28.294582165730386 usec\nrounds: 7637"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1498.887934794353,
            "unit": "iter/sec",
            "range": "stddev: 0.00004021803441680902",
            "extra": "mean: 667.1612845674148 usec\nrounds: 1121"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1332.6437369284308,
            "unit": "iter/sec",
            "range": "stddev: 0.000021383142634632555",
            "extra": "mean: 750.3880987013596 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2606.2499509429763,
            "unit": "iter/sec",
            "range": "stddev: 0.000042250888337268156",
            "extra": "mean: 383.69305278574166 usec\nrounds: 2046"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1487.8450898301514,
            "unit": "iter/sec",
            "range": "stddev: 0.000028614216540850468",
            "extra": "mean: 672.1129819463647 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5768.731979354455,
            "unit": "iter/sec",
            "range": "stddev: 0.000013669963307388441",
            "extra": "mean: 173.34832049380532 usec\nrounds: 4050"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 112.04362079634186,
            "unit": "iter/sec",
            "range": "stddev: 0.00006988270942951223",
            "extra": "mean: 8.925095359223247 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5925.960340823212,
            "unit": "iter/sec",
            "range": "stddev: 0.00001815288605837742",
            "extra": "mean: 168.74901998771793 usec\nrounds: 3252"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11355.882444677001,
            "unit": "iter/sec",
            "range": "stddev: 0.000004287405661362746",
            "extra": "mean: 88.06008734871536 usec\nrounds: 8094"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29753.241940317712,
            "unit": "iter/sec",
            "range": "stddev: 0.000005139247737721964",
            "extra": "mean: 33.60978282655412 usec\nrounds: 5788"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13716.307175246739,
            "unit": "iter/sec",
            "range": "stddev: 0.000008571617106621318",
            "extra": "mean: 72.90592046557978 usec\nrounds: 12372"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14361.744445536558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036277093665202234",
            "extra": "mean: 69.62942446109231 usec\nrounds: 12199"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5343.731226347963,
            "unit": "iter/sec",
            "range": "stddev: 0.000008336105082167436",
            "extra": "mean: 187.13516036685562 usec\nrounds: 3598"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 78599.41546576684,
            "unit": "iter/sec",
            "range": "stddev: 0.000002004171874548304",
            "extra": "mean: 12.722740927196076 usec\nrounds: 10526"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 334027.651686433,
            "unit": "iter/sec",
            "range": "stddev: 6.497603795373891e-7",
            "extra": "mean: 2.9937641238718333 usec\nrounds: 79847"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1460.6589325154919,
            "unit": "iter/sec",
            "range": "stddev: 0.000015067003336427705",
            "extra": "mean: 684.6225205208157 usec\nrounds: 999"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 195.79722864516452,
            "unit": "iter/sec",
            "range": "stddev: 0.002783829246428966",
            "extra": "mean: 5.107324587378405 msec\nrounds: 206"
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
          "id": "c06ea3190c58ce61b82c9da716d86f68907ba16c",
          "message": "refactor(io)!: drop HDF5/h5py; require zarr for MolStore\n\nrefactor(io)!: drop HDF5/h5py; require zarr for MolStore",
          "timestamp": "2026-07-17T15:24:19+08:00",
          "tree_id": "c9ebc1a4d7a58edc122d50f045871b28ede95054",
          "url": "https://github.com/MolCrafts/molpy/commit/c06ea3190c58ce61b82c9da716d86f68907ba16c"
        },
        "date": 1784273122471,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7815.759161437394,
            "unit": "iter/sec",
            "range": "stddev: 0.00000524100364720145",
            "extra": "mean: 127.94662416594862 usec\nrounds: 4196"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 112583.50909627824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011361406846125272",
            "extra": "mean: 8.882295533574355 usec\nrounds: 36764"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48868.34384717401,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022646310466120394",
            "extra": "mean: 20.46314487610426 usec\nrounds: 16559"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 55.62475899047825,
            "unit": "iter/sec",
            "range": "stddev: 0.0010422037665655897",
            "extra": "mean: 17.97760598245789 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33556.74674413869,
            "unit": "iter/sec",
            "range": "stddev: 0.000007210936151911008",
            "extra": "mean: 29.800266623720564 usec\nrounds: 7715"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18874.21893021785,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035774770690870565",
            "extra": "mean: 52.982324921482615 usec\nrounds: 10501"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.66556994559613,
            "unit": "iter/sec",
            "range": "stddev: 0.000017357425752487596",
            "extra": "mean: 2.7347392869084737 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3487.0571433077,
            "unit": "iter/sec",
            "range": "stddev: 0.000017269689510326626",
            "extra": "mean: 286.77476706086765 usec\nrounds: 2198"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 457982.97747420776,
            "unit": "iter/sec",
            "range": "stddev: 5.316035044864468e-7",
            "extra": "mean: 2.1834872673980925 usec\nrounds: 66011"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 801.1130494888903,
            "unit": "iter/sec",
            "range": "stddev: 0.00004680915615690912",
            "extra": "mean: 1.2482632764976171 msec\nrounds: 651"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52159.464567609706,
            "unit": "iter/sec",
            "range": "stddev: 0.000003426608545040644",
            "extra": "mean: 19.171975945109413 usec\nrounds: 11349"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 432.23573587945884,
            "unit": "iter/sec",
            "range": "stddev: 0.00011822405982995283",
            "extra": "mean: 2.3135523442209744 msec\nrounds: 398"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 213888.38363352756,
            "unit": "iter/sec",
            "range": "stddev: 8.454073566901016e-7",
            "extra": "mean: 4.675335719556335 usec\nrounds: 47215"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 123172.64931274451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010949072178310074",
            "extra": "mean: 8.118685483990244 usec\nrounds: 25792"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 149954.62290714576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010097905770319303",
            "extra": "mean: 6.668684036631639 usec\nrounds: 41144"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75086.95359381697,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013152727288382635",
            "extra": "mean: 13.31789281809863 usec\nrounds: 26469"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3541.3577934939785,
            "unit": "iter/sec",
            "range": "stddev: 0.000008230126160725627",
            "extra": "mean: 282.3775676767692 usec\nrounds: 2475"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15213.595587195743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035014955349754554",
            "extra": "mean: 65.73068110484233 usec\nrounds: 10897"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17050.503334357873,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033076562614201793",
            "extra": "mean: 58.64929500262523 usec\nrounds: 9705"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.40433062187606,
            "unit": "iter/sec",
            "range": "stddev: 0.000060116298161650736",
            "extra": "mean: 12.754397519478141 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1265.6843688129948,
            "unit": "iter/sec",
            "range": "stddev: 0.000009907077381481021",
            "extra": "mean: 790.0863948709714 usec\nrounds: 1170"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.72817596536598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000440205223725387",
            "extra": "mean: 7.260678455884484 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44537.130829965856,
            "unit": "iter/sec",
            "range": "stddev: 0.000001990819063372307",
            "extra": "mean: 22.453175167879728 usec\nrounds: 25016"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1085.2791896088886,
            "unit": "iter/sec",
            "range": "stddev: 0.00008145568825785487",
            "extra": "mean: 921.4218880953375 usec\nrounds: 840"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1278.15531458506,
            "unit": "iter/sec",
            "range": "stddev: 0.000015179753598019186",
            "extra": "mean: 782.377531579282 usec\nrounds: 950"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 104296.1459958864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017959130969758195",
            "extra": "mean: 9.58808199911281 usec\nrounds: 47781"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93590.59635606442,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014439392248088354",
            "extra": "mean: 10.684834149314646 usec\nrounds: 46361"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 79700.73219766752,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013997238187867423",
            "extra": "mean: 12.546936175189437 usec\nrounds: 46424"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 104164.71693818901,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011140065642594465",
            "extra": "mean: 9.600179690339836 usec\nrounds: 55885"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40067.139047328375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024122541897185553",
            "extra": "mean: 24.95810840945677 usec\nrounds: 15340"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 178212.644585257,
            "unit": "iter/sec",
            "range": "stddev: 0.000001026563377215455",
            "extra": "mean: 5.611274117654427 usec\nrounds: 16409"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 178597.69190772477,
            "unit": "iter/sec",
            "range": "stddev: 8.988527476994239e-7",
            "extra": "mean: 5.599176502889328 usec\nrounds: 27399"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 177914.61452107003,
            "unit": "iter/sec",
            "range": "stddev: 8.62942806694961e-7",
            "extra": "mean: 5.620673729878285 usec\nrounds: 27183"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 160020.86241394727,
            "unit": "iter/sec",
            "range": "stddev: 9.748256733959596e-7",
            "extra": "mean: 6.2491851682011745 usec\nrounds: 32619"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 167531.81218526987,
            "unit": "iter/sec",
            "range": "stddev: 9.405443295869136e-7",
            "extra": "mean: 5.969015597432453 usec\nrounds: 35647"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 177245.60859490946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010009888300375974",
            "extra": "mean: 5.6418887211218625 usec\nrounds: 28469"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.335999704660818,
            "unit": "iter/sec",
            "range": "stddev: 0.0003262165273341157",
            "extra": "mean: 42.85224599999774 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 849.6644146761003,
            "unit": "iter/sec",
            "range": "stddev: 0.000024656449205068693",
            "extra": "mean: 1.176935249643483 msec\nrounds: 701"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4561.306033828044,
            "unit": "iter/sec",
            "range": "stddev: 0.000007350406471547989",
            "extra": "mean: 219.23545418432647 usec\nrounds: 4027"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34523.48952409209,
            "unit": "iter/sec",
            "range": "stddev: 0.000004166861497231969",
            "extra": "mean: 28.965785724011294 usec\nrounds: 7257"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1440.0896013572515,
            "unit": "iter/sec",
            "range": "stddev: 0.000027497263540346858",
            "extra": "mean: 694.4012366018911 usec\nrounds: 989"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1296.8180223593438,
            "unit": "iter/sec",
            "range": "stddev: 0.000027786143252437393",
            "extra": "mean: 771.1182160937792 usec\nrounds: 1106"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2608.396288124682,
            "unit": "iter/sec",
            "range": "stddev: 0.00002069186458178744",
            "extra": "mean: 383.37732826592634 usec\nrounds: 2105"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1488.6872792780716,
            "unit": "iter/sec",
            "range": "stddev: 0.000036990388869416285",
            "extra": "mean: 671.7327500003513 usec\nrounds: 1196"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5490.768952209131,
            "unit": "iter/sec",
            "range": "stddev: 0.00003092591631384329",
            "extra": "mean: 182.12385345365234 usec\nrounds: 3214"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 109.3536065334044,
            "unit": "iter/sec",
            "range": "stddev: 0.00042706565245991684",
            "extra": "mean: 9.144645811883017 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5992.624030458729,
            "unit": "iter/sec",
            "range": "stddev: 0.000008073104306697469",
            "extra": "mean: 166.871806894158 usec\nrounds: 3133"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10537.870781693748,
            "unit": "iter/sec",
            "range": "stddev: 0.000004911465725016883",
            "extra": "mean: 94.89583054455242 usec\nrounds: 7052"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29867.93988800803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050006846177302185",
            "extra": "mean: 33.48071556825048 usec\nrounds: 6571"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13939.860332662782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033664648365814506",
            "extra": "mean: 71.73673022080995 usec\nrounds: 12273"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14192.273061123085,
            "unit": "iter/sec",
            "range": "stddev: 0.000003243937992398547",
            "extra": "mean: 70.4608765412851 usec\nrounds: 13057"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5232.039162624891,
            "unit": "iter/sec",
            "range": "stddev: 0.000022258299000642434",
            "extra": "mean: 191.13006782202763 usec\nrounds: 3981"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 79034.57078523746,
            "unit": "iter/sec",
            "range": "stddev: 0.000002242222304274114",
            "extra": "mean: 12.652690968833424 usec\nrounds: 15856"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 314261.92668753245,
            "unit": "iter/sec",
            "range": "stddev: 9.058092899446012e-7",
            "extra": "mean: 3.182059024904694 usec\nrounds: 85896"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1428.3110402071434,
            "unit": "iter/sec",
            "range": "stddev: 0.00003558329513404992",
            "extra": "mean: 700.1276135588598 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 189.41616017086926,
            "unit": "iter/sec",
            "range": "stddev: 0.0032586475745531684",
            "extra": "mean: 5.279380592964803 msec\nrounds: 199"
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
          "id": "8776b49d974d30940da26aff331c4f490f52fd96",
          "message": "Merge pull request #44 from Roy-Kid/release/v0.8.2\n\nrelease: v0.8.2",
          "timestamp": "2026-07-17T17:11:35+08:00",
          "tree_id": "2439dc5a82dbe8e057de90219f9a4373e9efc042",
          "url": "https://github.com/MolCrafts/molpy/commit/8776b49d974d30940da26aff331c4f490f52fd96"
        },
        "date": 1784279561029,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7725.272720970884,
            "unit": "iter/sec",
            "range": "stddev: 0.000004555581558374069",
            "extra": "mean: 129.44526829265436 usec\nrounds: 4879"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 112022.76269433054,
            "unit": "iter/sec",
            "range": "stddev: 0.000001122598295021847",
            "extra": "mean: 8.9267571692428 usec\nrounds: 40102"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 47618.45445569184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018723624311513013",
            "extra": "mean: 21.000261588298354 usec\nrounds: 18208"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.54794174638738,
            "unit": "iter/sec",
            "range": "stddev: 0.0009627952274592377",
            "extra": "mean: 16.515838047618942 msec\nrounds: 63"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 35766.03141130685,
            "unit": "iter/sec",
            "range": "stddev: 0.00000599869581339831",
            "extra": "mean: 27.959490067546778 usec\nrounds: 9766"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18901.90700406949,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030281448477088196",
            "extra": "mean: 52.90471484092609 usec\nrounds: 11320"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 364.87361467591643,
            "unit": "iter/sec",
            "range": "stddev: 0.000059153141599146",
            "extra": "mean: 2.740675016712863 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3511.46505627299,
            "unit": "iter/sec",
            "range": "stddev: 0.00004522468106565608",
            "extra": "mean: 284.78141857444064 usec\nrounds: 2315"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 482457.4229408231,
            "unit": "iter/sec",
            "range": "stddev: 5.214939652674952e-7",
            "extra": "mean: 2.072721762481116 usec\nrounds: 73714"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 818.2178561956578,
            "unit": "iter/sec",
            "range": "stddev: 0.00002380231079544003",
            "extra": "mean: 1.222168390029456 msec\nrounds: 682"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52476.24980384121,
            "unit": "iter/sec",
            "range": "stddev: 0.000002577743953927606",
            "extra": "mean: 19.056239798728928 usec\nrounds: 12719"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 444.85887583705914,
            "unit": "iter/sec",
            "range": "stddev: 0.00002902160814687819",
            "extra": "mean: 2.2479038956306567 msec\nrounds: 412"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217190.13931875557,
            "unit": "iter/sec",
            "range": "stddev: 7.638443103252287e-7",
            "extra": "mean: 4.604260594595256 usec\nrounds: 47595"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 123193.64628678831,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011614265659699916",
            "extra": "mean: 8.117301745189462 usec\nrounds: 30254"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 152040.89122134526,
            "unit": "iter/sec",
            "range": "stddev: 9.098036154142959e-7",
            "extra": "mean: 6.577177968157086 usec\nrounds: 53459"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 74745.5823785502,
            "unit": "iter/sec",
            "range": "stddev: 0.000001293453441950228",
            "extra": "mean: 13.378717085051047 usec\nrounds: 28323"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3505.2609948665063,
            "unit": "iter/sec",
            "range": "stddev: 0.00001817337814691604",
            "extra": "mean: 285.28546132927363 usec\nrounds: 2573"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15301.548699401672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037228587755535153",
            "extra": "mean: 65.35286196482207 usec\nrounds: 11258"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16387.501471152114,
            "unit": "iter/sec",
            "range": "stddev: 0.000009046851671259291",
            "extra": "mean: 61.02211504056058 usec\nrounds: 12074"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.25571092905568,
            "unit": "iter/sec",
            "range": "stddev: 0.00039968341825332413",
            "extra": "mean: 12.778620092105106 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1249.2305230246384,
            "unit": "iter/sec",
            "range": "stddev: 0.000012478122527556262",
            "extra": "mean: 800.4927686035071 usec\nrounds: 981"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.24472671449294,
            "unit": "iter/sec",
            "range": "stddev: 0.000028544572546361995",
            "extra": "mean: 7.286254444444173 msec\nrounds: 135"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44209.87566947273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031473921819194763",
            "extra": "mean: 22.619380508471053 usec\nrounds: 25960"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1105.5030777529637,
            "unit": "iter/sec",
            "range": "stddev: 0.00009483686895418284",
            "extra": "mean: 904.5655504031627 usec\nrounds: 992"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1271.4462547795297,
            "unit": "iter/sec",
            "range": "stddev: 0.000010501765858520826",
            "extra": "mean: 786.5059150088898 usec\nrounds: 1106"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 100867.99582500863,
            "unit": "iter/sec",
            "range": "stddev: 0.000001353661771468985",
            "extra": "mean: 9.913947350900628 usec\nrounds: 54037"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 87762.6242445881,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017869776569127628",
            "extra": "mean: 11.394372132869137 usec\nrounds: 47827"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 71592.64372858916,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019956818868453265",
            "extra": "mean: 13.967915527621019 usec\nrounds: 48359"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 98928.5421231623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012697625627894373",
            "extra": "mean: 10.108306243460433 usec\nrounds: 55482"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39461.15999732778,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019279495702517922",
            "extra": "mean: 25.34137364607928 usec\nrounds: 16157"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 181758.85698315082,
            "unit": "iter/sec",
            "range": "stddev: 8.556440660237078e-7",
            "extra": "mean: 5.501795161996979 usec\nrounds: 28276"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 181751.33347400607,
            "unit": "iter/sec",
            "range": "stddev: 9.132292156942308e-7",
            "extra": "mean: 5.502022906164918 usec\nrounds: 37981"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 179969.00704337098,
            "unit": "iter/sec",
            "range": "stddev: 9.843959411666417e-7",
            "extra": "mean: 5.556512293025035 usec\nrounds: 37623"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 159864.99700397803,
            "unit": "iter/sec",
            "range": "stddev: 9.099010672023152e-7",
            "extra": "mean: 6.255278007950148 usec\nrounds: 46103"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 162804.64427012953,
            "unit": "iter/sec",
            "range": "stddev: 9.378345338740224e-7",
            "extra": "mean: 6.142330917420114 usec\nrounds: 33528"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 171679.33936059455,
            "unit": "iter/sec",
            "range": "stddev: 8.80047357291131e-7",
            "extra": "mean: 5.8248127219292485 usec\nrounds: 41676"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.430051782319246,
            "unit": "iter/sec",
            "range": "stddev: 0.000303508823765907",
            "extra": "mean: 42.680230043478545 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 854.3796197600523,
            "unit": "iter/sec",
            "range": "stddev: 0.000024922232654051606",
            "extra": "mean: 1.1704399038460729 msec\nrounds: 676"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4550.96466265126,
            "unit": "iter/sec",
            "range": "stddev: 0.000005708341015120886",
            "extra": "mean: 219.73363322435225 usec\nrounds: 4286"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 35837.48977165714,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030944166824142064",
            "extra": "mean: 27.903740088148464 usec\nrounds: 6129"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1462.3141270807837,
            "unit": "iter/sec",
            "range": "stddev: 0.00004372914039050442",
            "extra": "mean: 683.847595725755 usec\nrounds: 1123"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1325.7510200466045,
            "unit": "iter/sec",
            "range": "stddev: 0.000020861186425748058",
            "extra": "mean: 754.2894441558467 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2645.664619132213,
            "unit": "iter/sec",
            "range": "stddev: 0.000020339880919154957",
            "extra": "mean: 377.97685797680714 usec\nrounds: 2056"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1528.824527397947,
            "unit": "iter/sec",
            "range": "stddev: 0.000021717188850999772",
            "extra": "mean: 654.0973029141519 usec\nrounds: 1304"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5800.531717839825,
            "unit": "iter/sec",
            "range": "stddev: 0.00001222333434696507",
            "extra": "mean: 172.3979884334483 usec\nrounds: 3977"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 111.92964952896354,
            "unit": "iter/sec",
            "range": "stddev: 0.00023823820832817897",
            "extra": "mean: 8.934183249999673 msec\nrounds: 92"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5857.397124630161,
            "unit": "iter/sec",
            "range": "stddev: 0.000007450568827286839",
            "extra": "mean: 170.7242959155071 usec\nrounds: 3403"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11793.739222851991,
            "unit": "iter/sec",
            "range": "stddev: 0.000005648967474447337",
            "extra": "mean: 84.79075050789342 usec\nrounds: 10337"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29553.051660464138,
            "unit": "iter/sec",
            "range": "stddev: 0.000004439812188194694",
            "extra": "mean: 33.83745311614614 usec\nrounds: 5215"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13652.229208482446,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037912399234437113",
            "extra": "mean: 73.24811096627919 usec\nrounds: 12265"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14491.28942411509,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029128069996061376",
            "extra": "mean: 69.00697175614275 usec\nrounds: 13348"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5314.09716238,
            "unit": "iter/sec",
            "range": "stddev: 0.0000080786354126954",
            "extra": "mean: 188.17871962885505 usec\nrounds: 3449"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 81045.74259528649,
            "unit": "iter/sec",
            "range": "stddev: 0.000001963883473880014",
            "extra": "mean: 12.338711053505218 usec\nrounds: 15425"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 336292.2395512413,
            "unit": "iter/sec",
            "range": "stddev: 6.165805298794916e-7",
            "extra": "mean: 2.973604152550266 usec\nrounds: 83949"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1430.9304396953216,
            "unit": "iter/sec",
            "range": "stddev: 0.00001351654844038335",
            "extra": "mean: 698.8459901747029 usec\nrounds: 916"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 192.52249174423136,
            "unit": "iter/sec",
            "range": "stddev: 0.002892346443543358",
            "extra": "mean: 5.194198303481928 msec\nrounds: 201"
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
          "id": "849b57d6f06b285eb900208dac2e0505c97fc82b",
          "message": "Merge pull request #45 from Roy-Kid/release/v0.9.0\n\nrelease: v0.9.0",
          "timestamp": "2026-07-21T15:52:34+02:00",
          "tree_id": "3e37ffe5b4f1409e3af320260d658e091d77c751",
          "url": "https://github.com/MolCrafts/molpy/commit/849b57d6f06b285eb900208dac2e0505c97fc82b"
        },
        "date": 1784642018688,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7794.936682996612,
            "unit": "iter/sec",
            "range": "stddev: 0.00000509837305995842",
            "extra": "mean: 128.28840575207462 usec\nrounds: 4520"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 114112.15131135582,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010170319907907414",
            "extra": "mean: 8.763308626716649 usec\nrounds: 36468"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48004.8159085931,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018006383016238481",
            "extra": "mean: 20.83124330492423 usec\nrounds: 18409"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 57.22610175207466,
            "unit": "iter/sec",
            "range": "stddev: 0.0010985724857398627",
            "extra": "mean: 17.4745434230761 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 34633.714797422515,
            "unit": "iter/sec",
            "range": "stddev: 0.000006362451058672884",
            "extra": "mean: 28.87359920381458 usec\nrounds: 9294"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18764.304083344156,
            "unit": "iter/sec",
            "range": "stddev: 0.000003087915999193112",
            "extra": "mean: 53.29267717888001 usec\nrounds: 11669"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 362.5237852877101,
            "unit": "iter/sec",
            "range": "stddev: 0.000013043162067370282",
            "extra": "mean: 2.7584396957743587 msec\nrounds: 355"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3668.215686621873,
            "unit": "iter/sec",
            "range": "stddev: 0.000013024038335753352",
            "extra": "mean: 272.612105020716 usec\nrounds: 2390"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 481716.7728776985,
            "unit": "iter/sec",
            "range": "stddev: 5.133610819342158e-7",
            "extra": "mean: 2.075908617476948 usec\nrounds: 66632"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 848.1349249571227,
            "unit": "iter/sec",
            "range": "stddev: 0.000023260523843806787",
            "extra": "mean: 1.1790576835997582 msec\nrounds: 689"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 54576.7818094896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026142817869892255",
            "extra": "mean: 18.322809935746776 usec\nrounds: 12601"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 452.83263578137735,
            "unit": "iter/sec",
            "range": "stddev: 0.000029631806357835035",
            "extra": "mean: 2.208321399526489 msec\nrounds: 423"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 216119.65608728502,
            "unit": "iter/sec",
            "range": "stddev: 7.992071317988682e-7",
            "extra": "mean: 4.627066404344667 usec\nrounds: 47015"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 122198.85096953675,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012109914344157045",
            "extra": "mean: 8.183383002916226 usec\nrounds: 27287"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 147897.3747784401,
            "unit": "iter/sec",
            "range": "stddev: 0.000001057946157655858",
            "extra": "mean: 6.761445235238726 usec\nrounds: 45102"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75343.86741029182,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012961411193418008",
            "extra": "mean: 13.27248035403346 usec\nrounds: 30617"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3498.81292715577,
            "unit": "iter/sec",
            "range": "stddev: 0.000013805893388566878",
            "extra": "mean: 285.8112224973722 usec\nrounds: 2827"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15532.218237128507,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032032340958203753",
            "extra": "mean: 64.38230423582262 usec\nrounds: 11261"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16796.84211226805,
            "unit": "iter/sec",
            "range": "stddev: 0.000003165825531215036",
            "extra": "mean: 59.535000288513864 usec\nrounds: 10396"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.8850775541142,
            "unit": "iter/sec",
            "range": "stddev: 0.0006644905513688845",
            "extra": "mean: 12.67666878205212 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1264.40776496009,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146942743030609",
            "extra": "mean: 790.8841021959116 usec\nrounds: 1184"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 138.90361575335538,
            "unit": "iter/sec",
            "range": "stddev: 0.000027623018624221383",
            "extra": "mean: 7.199236640287701 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44593.65078592133,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018935006043788791",
            "extra": "mean: 22.424717025314962 usec\nrounds: 25186"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1028.294252806643,
            "unit": "iter/sec",
            "range": "stddev: 0.00009360912356010176",
            "extra": "mean: 972.4842838230242 usec\nrounds: 680"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1259.055214293725,
            "unit": "iter/sec",
            "range": "stddev: 0.000012500822446836875",
            "extra": "mean: 794.2463433273308 usec\nrounds: 1034"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 104017.47464236789,
            "unit": "iter/sec",
            "range": "stddev: 0.000002082894045111961",
            "extra": "mean: 9.613769257888567 usec\nrounds: 53926"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93796.15855295476,
            "unit": "iter/sec",
            "range": "stddev: 0.000001158602172773984",
            "extra": "mean: 10.661417433587403 usec\nrounds: 40313"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 78449.2606005579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013222274433611742",
            "extra": "mean: 12.747092736689076 usec\nrounds: 21836"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 103260.25691307301,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010408181246172104",
            "extra": "mean: 9.684267983584665 usec\nrounds: 59485"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39170.83035108116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024631178615958945",
            "extra": "mean: 25.52920096503389 usec\nrounds: 15958"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 178294.0143540436,
            "unit": "iter/sec",
            "range": "stddev: 9.808467777967494e-7",
            "extra": "mean: 5.6087132460558715 usec\nrounds: 18413"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 178916.06884197533,
            "unit": "iter/sec",
            "range": "stddev: 8.4355542679172e-7",
            "extra": "mean: 5.589212900062283 usec\nrounds: 50310"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 178190.70494710503,
            "unit": "iter/sec",
            "range": "stddev: 8.671767317577996e-7",
            "extra": "mean: 5.6119650028706305 usec\nrounds: 46861"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 163513.367584137,
            "unit": "iter/sec",
            "range": "stddev: 8.992154144293541e-7",
            "extra": "mean: 6.115707937367522 usec\nrounds: 39358"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 169081.82497347356,
            "unit": "iter/sec",
            "range": "stddev: 9.607699794835415e-7",
            "extra": "mean: 5.914296229987376 usec\nrounds: 43078"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180832.99403583293,
            "unit": "iter/sec",
            "range": "stddev: 8.954252413754661e-7",
            "extra": "mean: 5.529964292920158 usec\nrounds: 45145"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.385671300443303,
            "unit": "iter/sec",
            "range": "stddev: 0.00039680194751896277",
            "extra": "mean: 42.761227041664775 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 852.1444953869691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179476687010246",
            "extra": "mean: 1.173509898160978 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4506.9552469664095,
            "unit": "iter/sec",
            "range": "stddev: 0.000005491042431957679",
            "extra": "mean: 221.87928328622542 usec\nrounds: 3883"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 35386.59569569823,
            "unit": "iter/sec",
            "range": "stddev: 0.00000354544758734483",
            "extra": "mean: 28.259288025311935 usec\nrounds: 6614"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1504.471426784777,
            "unit": "iter/sec",
            "range": "stddev: 0.000018550386786761193",
            "extra": "mean: 664.6852723132879 usec\nrounds: 1098"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1350.1950087339592,
            "unit": "iter/sec",
            "range": "stddev: 0.000020473440516367482",
            "extra": "mean: 740.6337555177844 usec\nrounds: 1178"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2704.2493957807637,
            "unit": "iter/sec",
            "range": "stddev: 0.00001683290097537934",
            "extra": "mean: 369.7883788234268 usec\nrounds: 2125"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1526.2544567240345,
            "unit": "iter/sec",
            "range": "stddev: 0.00002268783798343029",
            "extra": "mean: 655.1987419885465 usec\nrounds: 1217"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5807.923433967625,
            "unit": "iter/sec",
            "range": "stddev: 0.000013180282657924948",
            "extra": "mean: 172.17857834548965 usec\nrounds: 4110"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 113.17685325061701,
            "unit": "iter/sec",
            "range": "stddev: 0.0001967336351349379",
            "extra": "mean: 8.83572896116502 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 6086.7962833601205,
            "unit": "iter/sec",
            "range": "stddev: 0.000017547005021426823",
            "extra": "mean: 164.2900392007149 usec\nrounds: 2602"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11840.863239271242,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044729851718770635",
            "extra": "mean: 84.45330207711663 usec\nrounds: 6644"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 30578.45493101921,
            "unit": "iter/sec",
            "range": "stddev: 0.000004517683590607238",
            "extra": "mean: 32.70276416044769 usec\nrounds: 7980"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13700.608081692366,
            "unit": "iter/sec",
            "range": "stddev: 0.000003424152831655617",
            "extra": "mean: 72.98946105437936 usec\nrounds: 11760"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14437.84302492613,
            "unit": "iter/sec",
            "range": "stddev: 0.000004289514166539662",
            "extra": "mean: 69.26242363721201 usec\nrounds: 13318"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5325.795151876032,
            "unit": "iter/sec",
            "range": "stddev: 0.000008825249055270472",
            "extra": "mean: 187.76538929548317 usec\nrounds: 3961"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 80181.99640439276,
            "unit": "iter/sec",
            "range": "stddev: 0.000002070068400854923",
            "extra": "mean: 12.471627607730843 usec\nrounds: 11888"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 343473.66730853665,
            "unit": "iter/sec",
            "range": "stddev: 7.832572908918086e-7",
            "extra": "mean: 2.9114313415523547 usec\nrounds: 78592"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1397.3254511646694,
            "unit": "iter/sec",
            "range": "stddev: 0.00008020394817079792",
            "extra": "mean: 715.6528918631668 usec\nrounds: 934"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 194.0299938960468,
            "unit": "iter/sec",
            "range": "stddev: 0.002814638356011318",
            "extra": "mean: 5.153842351485917 msec\nrounds: 202"
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
          "id": "973a0195aac207d9149877924b977c1591636f29",
          "message": "Merge pull request #46 from Roy-Kid/release/v0.9.1\n\nrelease: v0.9.1",
          "timestamp": "2026-07-22T19:28:25+02:00",
          "tree_id": "1c9038e3675b1d3f84caf9b21eff118cbf025616",
          "url": "https://github.com/MolCrafts/molpy/commit/973a0195aac207d9149877924b977c1591636f29"
        },
        "date": 1784741369594,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 11429.018740161733,
            "unit": "iter/sec",
            "range": "stddev: 0.000004232248814321028",
            "extra": "mean: 87.49657540467459 usec\nrounds: 5066"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113470.06881054226,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010469239515427805",
            "extra": "mean: 8.812896744335914 usec\nrounds: 32221"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48470.31567965951,
            "unit": "iter/sec",
            "range": "stddev: 0.000001608241831196402",
            "extra": "mean: 20.631183972660786 usec\nrounds: 16921"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 57.60317964466687,
            "unit": "iter/sec",
            "range": "stddev: 0.0010122275032674133",
            "extra": "mean: 17.360152793103392 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 35043.59435772367,
            "unit": "iter/sec",
            "range": "stddev: 0.000005965939390323437",
            "extra": "mean: 28.535885611277152 usec\nrounds: 12204"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 17787.96831759317,
            "unit": "iter/sec",
            "range": "stddev: 0.000002882943293213185",
            "extra": "mean: 56.21777496707991 usec\nrounds: 12136"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 384.81346815291886,
            "unit": "iter/sec",
            "range": "stddev: 0.00015136610548749382",
            "extra": "mean: 2.5986616445623354 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 4062.21614005646,
            "unit": "iter/sec",
            "range": "stddev: 0.000011196436343843283",
            "extra": "mean: 246.17104691679484 usec\nrounds: 2238"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 457880.124630735,
            "unit": "iter/sec",
            "range": "stddev: 4.6289158375073383e-7",
            "extra": "mean: 2.183977740476214 usec\nrounds: 66129"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1052.598197380448,
            "unit": "iter/sec",
            "range": "stddev: 0.000032847426733754926",
            "extra": "mean: 950.0301278195739 usec\nrounds: 798"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 58269.783000844865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017218503943442478",
            "extra": "mean: 17.161553527417475 usec\nrounds: 11779"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 563.2359533350111,
            "unit": "iter/sec",
            "range": "stddev: 0.00004315920060356089",
            "extra": "mean: 1.7754548410463473 msec\nrounds: 497"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 215054.09517393337,
            "unit": "iter/sec",
            "range": "stddev: 8.026705412437599e-7",
            "extra": "mean: 4.64999282711269 usec\nrounds: 19518"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 116538.58305227416,
            "unit": "iter/sec",
            "range": "stddev: 9.254174295267313e-7",
            "extra": "mean: 8.580849138619124 usec\nrounds: 25076"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 144590.19656708903,
            "unit": "iter/sec",
            "range": "stddev: 8.281475753943322e-7",
            "extra": "mean: 6.916098212343225 usec\nrounds: 37480"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 70411.66218876802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012259549481560193",
            "extra": "mean: 14.202192774814494 usec\nrounds: 25356"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3925.47485552963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000202278206742115",
            "extra": "mean: 254.74625027628122 usec\nrounds: 2713"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 14659.380743909896,
            "unit": "iter/sec",
            "range": "stddev: 0.000010329511201183446",
            "extra": "mean: 68.21570552463075 usec\nrounds: 7240"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16644.96365123042,
            "unit": "iter/sec",
            "range": "stddev: 0.000004543442351108101",
            "extra": "mean: 60.07823272873765 usec\nrounds: 12680"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.48922093078649,
            "unit": "iter/sec",
            "range": "stddev: 0.0000443174249724798",
            "extra": "mean: 12.580322064934165 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1133.5339068568037,
            "unit": "iter/sec",
            "range": "stddev: 0.000019026643577767095",
            "extra": "mean: 882.1968129501461 usec\nrounds: 834"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 140.95543700145345,
            "unit": "iter/sec",
            "range": "stddev: 0.000036292931212653094",
            "extra": "mean: 7.094440776978958 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 41535.3772050934,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016966281310003499",
            "extra": "mean: 24.075861766276965 usec\nrounds: 22896"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1077.3887350100613,
            "unit": "iter/sec",
            "range": "stddev: 0.00010373796356700054",
            "extra": "mean: 928.1700907988995 usec\nrounds: 826"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1105.237613435245,
            "unit": "iter/sec",
            "range": "stddev: 0.000011292498726980476",
            "extra": "mean: 904.7828157891308 usec\nrounds: 912"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 107618.8814238883,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010247431588218837",
            "extra": "mean: 9.292049747861704 usec\nrounds: 45992"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 95917.14734165698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010866272441253078",
            "extra": "mean: 10.425664521047514 usec\nrounds: 16779"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 82148.07118814164,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012117929275800251",
            "extra": "mean: 12.173140349329998 usec\nrounds: 46484"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 108542.82777478002,
            "unit": "iter/sec",
            "range": "stddev: 9.894103673486216e-7",
            "extra": "mean: 9.21295326923803 usec\nrounds: 55167"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39758.36008755727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018250840043625882",
            "extra": "mean: 25.15194283158975 usec\nrounds: 13364"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 171262.18665317737,
            "unit": "iter/sec",
            "range": "stddev: 7.670885473189661e-7",
            "extra": "mean: 5.8390005379593655 usec\nrounds: 20449"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 171117.88732399093,
            "unit": "iter/sec",
            "range": "stddev: 7.403215068325688e-7",
            "extra": "mean: 5.843924417478469 usec\nrounds: 29094"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 169632.81511193723,
            "unit": "iter/sec",
            "range": "stddev: 8.139306678597153e-7",
            "extra": "mean: 5.895085802473538 usec\nrounds: 29463"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 156474.7766575253,
            "unit": "iter/sec",
            "range": "stddev: 7.902658112529021e-7",
            "extra": "mean: 6.390806373788214 usec\nrounds: 24444"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 166132.94945164426,
            "unit": "iter/sec",
            "range": "stddev: 8.173187157034051e-7",
            "extra": "mean: 6.019275545884813 usec\nrounds: 29995"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 175260.73854618412,
            "unit": "iter/sec",
            "range": "stddev: 7.960912498843877e-7",
            "extra": "mean: 5.705784468872835 usec\nrounds: 29077"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.770527647673706,
            "unit": "iter/sec",
            "range": "stddev: 0.0003493161048488131",
            "extra": "mean: 42.068902080003454 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 859.0483636954259,
            "unit": "iter/sec",
            "range": "stddev: 0.000010378011078475905",
            "extra": "mean: 1.1640788135585673 msec\nrounds: 649"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 5988.727764731207,
            "unit": "iter/sec",
            "range": "stddev: 0.000005937136897988157",
            "extra": "mean: 166.9803736762249 usec\nrounds: 5288"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34111.62012978251,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029596222637078387",
            "extra": "mean: 29.315523454921163 usec\nrounds: 6715"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1765.8381393881339,
            "unit": "iter/sec",
            "range": "stddev: 0.0000632264172506494",
            "extra": "mean: 566.3033194800639 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1633.1368216306316,
            "unit": "iter/sec",
            "range": "stddev: 0.000014707590132105268",
            "extra": "mean: 612.3185680190187 usec\nrounds: 1257"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3275.6855687296056,
            "unit": "iter/sec",
            "range": "stddev: 0.00001521654588437598",
            "extra": "mean: 305.27960606054916 usec\nrounds: 2376"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2163.8009469105627,
            "unit": "iter/sec",
            "range": "stddev: 0.00001762794644299398",
            "extra": "mean: 462.1497191910294 usec\nrounds: 1631"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 7668.070625054756,
            "unit": "iter/sec",
            "range": "stddev: 0.000008454683419366774",
            "extra": "mean: 130.41090111149822 usec\nrounds: 4318"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 106.2368002537371,
            "unit": "iter/sec",
            "range": "stddev: 0.00020528046451765783",
            "extra": "mean: 9.412934102039872 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 8000.319907130574,
            "unit": "iter/sec",
            "range": "stddev: 0.000010407598721406583",
            "extra": "mean: 124.9950016509607 usec\nrounds: 3028"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 12314.947261303063,
            "unit": "iter/sec",
            "range": "stddev: 0.000003705196460334099",
            "extra": "mean: 81.2021341855254 usec\nrounds: 9621"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 37916.462106022314,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022661350379956306",
            "extra": "mean: 26.37376866026667 usec\nrounds: 5801"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 11826.563018344703,
            "unit": "iter/sec",
            "range": "stddev: 0.000006966274059955359",
            "extra": "mean: 84.55541973173914 usec\nrounds: 8129"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 13478.036810018566,
            "unit": "iter/sec",
            "range": "stddev: 0.000003057343533260052",
            "extra": "mean: 74.1947817842933 usec\nrounds: 12341"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 4941.3400908322365,
            "unit": "iter/sec",
            "range": "stddev: 0.000007084994252775273",
            "extra": "mean: 202.3742510367419 usec\nrounds: 3617"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 84859.2638208747,
            "unit": "iter/sec",
            "range": "stddev: 0.000001160986180738325",
            "extra": "mean: 11.784217243633549 usec\nrounds: 13547"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 346768.17154373875,
            "unit": "iter/sec",
            "range": "stddev: 5.379913858266458e-7",
            "extra": "mean: 2.8837710091679147 usec\nrounds: 87200"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1335.6170026162479,
            "unit": "iter/sec",
            "range": "stddev: 0.000021968810399050462",
            "extra": "mean: 748.7176324059735 usec\nrounds: 827"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 198.21699657553768,
            "unit": "iter/sec",
            "range": "stddev: 0.003884684681909149",
            "extra": "mean: 5.044976047848219 msec\nrounds: 209"
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
          "id": "1c5078d6a1e0ac37269554f3935669a4c223a480",
          "message": "Merge pull request #47 from Roy-Kid/release/v0.9.2\n\nrelease: v0.9.2",
          "timestamp": "2026-07-23T14:06:34+02:00",
          "tree_id": "c067bd81d08e1ea82fb1e40c228b0163e6348ac5",
          "url": "https://github.com/MolCrafts/molpy/commit/1c5078d6a1e0ac37269554f3935669a4c223a480"
        },
        "date": 1784808464739,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 14698.534651238308,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028006246169383036",
            "extra": "mean: 68.03399275694144 usec\nrounds: 6489"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 142975.81629204433,
            "unit": "iter/sec",
            "range": "stddev: 6.863557872235669e-7",
            "extra": "mean: 6.994189828281075 usec\nrounds: 32856"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 62270.578252318024,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010927552061208537",
            "extra": "mean: 16.05894835194942 usec\nrounds: 20872"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 74.60736628481497,
            "unit": "iter/sec",
            "range": "stddev: 0.000726232723488843",
            "extra": "mean: 13.40350222500124 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 46066.6567012942,
            "unit": "iter/sec",
            "range": "stddev: 0.000003989956616438562",
            "extra": "mean: 21.707674739328453 usec\nrounds: 13048"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 23011.86925193399,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025146264189826416",
            "extra": "mean: 43.45583529316972 usec\nrounds: 12246"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 499.1162493324358,
            "unit": "iter/sec",
            "range": "stddev: 0.00006507855392043567",
            "extra": "mean: 2.003541261855314 msec\nrounds: 485"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 5330.194918589106,
            "unit": "iter/sec",
            "range": "stddev: 0.000013315173607948359",
            "extra": "mean: 187.61039985845366 usec\nrounds: 2826"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 590698.0302967449,
            "unit": "iter/sec",
            "range": "stddev: 3.311747159285042e-7",
            "extra": "mean: 1.6929123658963903 usec\nrounds: 79752"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1374.4449454045573,
            "unit": "iter/sec",
            "range": "stddev: 0.000018169940970338288",
            "extra": "mean: 727.5664284287923 usec\nrounds: 999"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 76551.18604857051,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012663508945188378",
            "extra": "mean: 13.063154885223016 usec\nrounds: 13991"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 721.2770950274283,
            "unit": "iter/sec",
            "range": "stddev: 0.00002202739184013356",
            "extra": "mean: 1.3864297187504235 msec\nrounds: 608"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 275346.79455113574,
            "unit": "iter/sec",
            "range": "stddev: 4.966062152375447e-7",
            "extra": "mean: 3.6317836989174976 usec\nrounds: 48377"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 150533.65428658982,
            "unit": "iter/sec",
            "range": "stddev: 6.502469643676421e-7",
            "extra": "mean: 6.643032780538062 usec\nrounds: 26479"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 187443.5990850089,
            "unit": "iter/sec",
            "range": "stddev: 5.867661055241534e-7",
            "extra": "mean: 5.3349381087506895 usec\nrounds: 40458"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 90809.81444277595,
            "unit": "iter/sec",
            "range": "stddev: 8.243586130399287e-7",
            "extra": "mean: 11.012025584857378 usec\nrounds: 27868"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 5117.879122159689,
            "unit": "iter/sec",
            "range": "stddev: 0.000006199808119755392",
            "extra": "mean: 195.39343859649637 usec\nrounds: 3249"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 19745.561694958516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022638143927611742",
            "extra": "mean: 50.644292395861406 usec\nrounds: 13598"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 21546.109136575364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027985514915832407",
            "extra": "mean: 46.412092023726956 usec\nrounds: 15822"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 100.12566177313202,
            "unit": "iter/sec",
            "range": "stddev: 0.00007646602127990267",
            "extra": "mean: 9.987449593749828 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1468.9688234828259,
            "unit": "iter/sec",
            "range": "stddev: 0.0000076230988422908116",
            "extra": "mean: 680.7496415268143 usec\nrounds: 1074"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 176.15936339697535,
            "unit": "iter/sec",
            "range": "stddev: 0.000020016634374800387",
            "extra": "mean: 5.676678098265483 msec\nrounds: 173"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 51210.273374154305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011651022994594826",
            "extra": "mean: 19.527331804963524 usec\nrounds: 27697"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1429.0071826135409,
            "unit": "iter/sec",
            "range": "stddev: 0.0000748251514391896",
            "extra": "mean: 699.7865456288885 usec\nrounds: 1041"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1435.7545647901661,
            "unit": "iter/sec",
            "range": "stddev: 0.000009288830247675616",
            "extra": "mean: 696.4978726333695 usec\nrounds: 1162"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 140220.14650539256,
            "unit": "iter/sec",
            "range": "stddev: 6.664224495605552e-7",
            "extra": "mean: 7.131642812550777 usec\nrounds: 60772"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 123343.6791919708,
            "unit": "iter/sec",
            "range": "stddev: 8.955495393449838e-7",
            "extra": "mean: 8.107428013750186 usec\nrounds: 45078"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 107043.0986652646,
            "unit": "iter/sec",
            "range": "stddev: 7.861045901783736e-7",
            "extra": "mean: 9.342031503844154 usec\nrounds: 65262"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 142112.13763327236,
            "unit": "iter/sec",
            "range": "stddev: 6.904389380187369e-7",
            "extra": "mean: 7.036696630238236 usec\nrounds: 78996"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 51342.32696696958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011982612603241082",
            "extra": "mean: 19.477107078596905 usec\nrounds: 17828"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 222206.77083229908,
            "unit": "iter/sec",
            "range": "stddev: 5.143071992424414e-7",
            "extra": "mean: 4.500312912403135 usec\nrounds: 27439"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 222344.30505953138,
            "unit": "iter/sec",
            "range": "stddev: 5.12017296178693e-7",
            "extra": "mean: 4.497529179945742 usec\nrounds: 46419"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 222033.086257719,
            "unit": "iter/sec",
            "range": "stddev: 5.143416208607332e-7",
            "extra": "mean: 4.503833265819116 usec\nrounds: 42007"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 207305.16854817222,
            "unit": "iter/sec",
            "range": "stddev: 5.360396585328005e-7",
            "extra": "mean: 4.82380640580906 usec\nrounds: 40838"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 214300.39746631068,
            "unit": "iter/sec",
            "range": "stddev: 6.150419112146127e-7",
            "extra": "mean: 4.666346921531987 usec\nrounds: 45039"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 229310.02363533998,
            "unit": "iter/sec",
            "range": "stddev: 4.931383752981119e-7",
            "extra": "mean: 4.360908363911074 usec\nrounds: 48518"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 28.37266542716802,
            "unit": "iter/sec",
            "range": "stddev: 0.0006660962543589054",
            "extra": "mean: 35.2451905714314 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 1109.7604394531227,
            "unit": "iter/sec",
            "range": "stddev: 0.00000827879410904714",
            "extra": "mean: 901.0953755864542 usec\nrounds: 852"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 8114.495920569992,
            "unit": "iter/sec",
            "range": "stddev: 0.000003886907009202012",
            "extra": "mean: 123.23624409805068 usec\nrounds: 7116"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 42607.59585172702,
            "unit": "iter/sec",
            "range": "stddev: 0.000002349371787859076",
            "extra": "mean: 23.46999355420019 usec\nrounds: 7912"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 2305.0782692019834,
            "unit": "iter/sec",
            "range": "stddev: 0.000011080569978003644",
            "extra": "mean: 433.824748322407 usec\nrounds: 1490"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 2076.5952079559706,
            "unit": "iter/sec",
            "range": "stddev: 0.000013791220951343632",
            "extra": "mean: 481.55750151437445 usec\nrounds: 1651"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 4207.832364993975,
            "unit": "iter/sec",
            "range": "stddev: 0.00000945340846830214",
            "extra": "mean: 237.6520529475589 usec\nrounds: 2833"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2776.9777102607463,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154075147225599",
            "extra": "mean: 360.10371862369186 usec\nrounds: 1976"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 9453.956817509015,
            "unit": "iter/sec",
            "range": "stddev: 0.000015081125524195718",
            "extra": "mean: 105.77581633840019 usec\nrounds: 5276"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 137.06863159058923,
            "unit": "iter/sec",
            "range": "stddev: 0.00012483321156072925",
            "extra": "mean: 7.295615257814081 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 10186.839049837503,
            "unit": "iter/sec",
            "range": "stddev: 0.000004696027588548921",
            "extra": "mean: 98.1658780616497 usec\nrounds: 3838"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 16201.564637811027,
            "unit": "iter/sec",
            "range": "stddev: 0.000005183312073948358",
            "extra": "mean: 61.72243374977571 usec\nrounds: 12800"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 48111.9804227981,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036951595928707462",
            "extra": "mean: 20.78484384164209 usec\nrounds: 6487"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 16178.784877031843,
            "unit": "iter/sec",
            "range": "stddev: 0.000004128993648531146",
            "extra": "mean: 61.80933905732603 usec\nrounds: 12753"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 16948.26832106979,
            "unit": "iter/sec",
            "range": "stddev: 0.000002323904806659034",
            "extra": "mean: 59.00307813493946 usec\nrounds: 15870"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 6346.426946865792,
            "unit": "iter/sec",
            "range": "stddev: 0.000010869521396292476",
            "extra": "mean: 157.56897674428507 usec\nrounds: 4300"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 108829.06173643943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013496706048178328",
            "extra": "mean: 9.18872205681406 usec\nrounds: 15129"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 438378.764792972,
            "unit": "iter/sec",
            "range": "stddev: 4.761572597972643e-7",
            "extra": "mean: 2.281132391237651 usec\nrounds: 97514"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1752.3646360710288,
            "unit": "iter/sec",
            "range": "stddev: 0.000014779233130144843",
            "extra": "mean: 570.6574872693715 usec\nrounds: 1139"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 251.25924144545726,
            "unit": "iter/sec",
            "range": "stddev: 0.003201919046401878",
            "extra": "mean: 3.97995311235976 msec\nrounds: 267"
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
          "id": "c27e46074a9c7d6e9ef0c4fb9fab05340247b3e5",
          "message": "Merge pull request #48 from Roy-Kid/release/v0.9.3\n\nrelease: v0.9.3",
          "timestamp": "2026-07-23T14:35:59+02:00",
          "tree_id": "48d8e5440453ad76602b28d3263f2ccb1a1a5a42",
          "url": "https://github.com/MolCrafts/molpy/commit/c27e46074a9c7d6e9ef0c4fb9fab05340247b3e5"
        },
        "date": 1784810228097,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 11173.030322056784,
            "unit": "iter/sec",
            "range": "stddev: 0.000004891896175929177",
            "extra": "mean: 89.50123387975513 usec\nrounds: 5490"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113301.07066518709,
            "unit": "iter/sec",
            "range": "stddev: 9.578719527662843e-7",
            "extra": "mean: 8.826041926426916 usec\nrounds: 35610"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 47970.674146766774,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016763353384864049",
            "extra": "mean: 20.846069349379786 usec\nrounds: 16799"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.32229471850864,
            "unit": "iter/sec",
            "range": "stddev: 0.0009322375553841047",
            "extra": "mean: 16.577618684210478 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 36328.83194557504,
            "unit": "iter/sec",
            "range": "stddev: 0.000005057384836627335",
            "extra": "mean: 27.526346057536898 usec\nrounds: 12784"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 17877.050279056908,
            "unit": "iter/sec",
            "range": "stddev: 0.000003338815294570964",
            "extra": "mean: 55.9376398449529 usec\nrounds: 10837"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 384.9804405405366,
            "unit": "iter/sec",
            "range": "stddev: 0.00022412833251862934",
            "extra": "mean: 2.5975345620051176 msec\nrounds: 379"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3964.9002816610514,
            "unit": "iter/sec",
            "range": "stddev: 0.000015154692423602904",
            "extra": "mean: 252.21315265489122 usec\nrounds: 2260"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 455685.06791457464,
            "unit": "iter/sec",
            "range": "stddev: 4.336053452471257e-7",
            "extra": "mean: 2.1944980654653925 usec\nrounds: 65907"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1047.206351627791,
            "unit": "iter/sec",
            "range": "stddev: 0.00010459704904465053",
            "extra": "mean: 954.9216335878665 usec\nrounds: 786"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 58524.58659972369,
            "unit": "iter/sec",
            "range": "stddev: 0.000002141209446731075",
            "extra": "mean: 17.086835774500305 usec\nrounds: 12221"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 561.9721848740276,
            "unit": "iter/sec",
            "range": "stddev: 0.00003297821313439409",
            "extra": "mean: 1.779447500990038 msec\nrounds: 505"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 218947.62876775762,
            "unit": "iter/sec",
            "range": "stddev: 6.820595423574429e-7",
            "extra": "mean: 4.567302261403896 usec\nrounds: 44928"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 117263.09335540031,
            "unit": "iter/sec",
            "range": "stddev: 9.812968594115703e-7",
            "extra": "mean: 8.527832341666153 usec\nrounds: 25546"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 146335.54328846218,
            "unit": "iter/sec",
            "range": "stddev: 7.967766662123042e-7",
            "extra": "mean: 6.833609781519463 usec\nrounds: 39319"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 71183.12079024922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010963409028965328",
            "extra": "mean: 14.048274210211105 usec\nrounds: 27096"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3965.6849684331423,
            "unit": "iter/sec",
            "range": "stddev: 0.000008116126399701867",
            "extra": "mean: 252.1632474490539 usec\nrounds: 2744"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15183.529153007827,
            "unit": "iter/sec",
            "range": "stddev: 0.00000376924870450206",
            "extra": "mean: 65.86084104181418 usec\nrounds: 8908"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16745.763397808896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033014672888208338",
            "extra": "mean: 59.716596744155915 usec\nrounds: 12347"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.64505010466752,
            "unit": "iter/sec",
            "range": "stddev: 0.000038966926813803003",
            "extra": "mean: 12.555708090908665 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1131.8280332288648,
            "unit": "iter/sec",
            "range": "stddev: 0.000008769247204872377",
            "extra": "mean: 883.5264462810773 usec\nrounds: 847"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 139.90508346429044,
            "unit": "iter/sec",
            "range": "stddev: 0.00002666467218694194",
            "extra": "mean: 7.147703108695412 msec\nrounds: 138"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 42369.61414589507,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014606675430548106",
            "extra": "mean: 23.60181984562358 usec\nrounds: 24096"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1084.2574297051983,
            "unit": "iter/sec",
            "range": "stddev: 0.00010386992388815334",
            "extra": "mean: 922.2901984373699 usec\nrounds: 640"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1106.7931833902394,
            "unit": "iter/sec",
            "range": "stddev: 0.000017040816799248507",
            "extra": "mean: 903.5111663200533 usec\nrounds: 962"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 103843.45543123133,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035606388477968554",
            "extra": "mean: 9.629879859518292 usec\nrounds: 52114"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 96440.66652238581,
            "unit": "iter/sec",
            "range": "stddev: 0.000001063650059241573",
            "extra": "mean: 10.369069771701337 usec\nrounds: 38067"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 83252.33509440055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012451930190392483",
            "extra": "mean: 12.011675094351304 usec\nrounds: 50073"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 110320.96529593105,
            "unit": "iter/sec",
            "range": "stddev: 9.7441536775849e-7",
            "extra": "mean: 9.064460207699822 usec\nrounds: 61444"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39683.24941712095,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021893941858302843",
            "extra": "mean: 25.199549298212453 usec\nrounds: 11187"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 172833.50486897715,
            "unit": "iter/sec",
            "range": "stddev: 7.746279490734509e-7",
            "extra": "mean: 5.785915183274719 usec\nrounds: 24606"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 174972.13866327482,
            "unit": "iter/sec",
            "range": "stddev: 7.465982303600926e-7",
            "extra": "mean: 5.715195617083074 usec\nrounds: 35866"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 173671.42821267023,
            "unit": "iter/sec",
            "range": "stddev: 7.368757311914196e-7",
            "extra": "mean: 5.757999518351659 usec\nrounds: 39451"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 160706.46802076703,
            "unit": "iter/sec",
            "range": "stddev: 0.000001032037041641993",
            "extra": "mean: 6.222524907154183 usec\nrounds: 36616"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 167226.67290863534,
            "unit": "iter/sec",
            "range": "stddev: 7.840027382387483e-7",
            "extra": "mean: 5.97990728755545 usec\nrounds: 37708"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 178610.6635570858,
            "unit": "iter/sec",
            "range": "stddev: 7.707763599007936e-7",
            "extra": "mean: 5.598769861130882 usec\nrounds: 37951"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.85247150961405,
            "unit": "iter/sec",
            "range": "stddev: 0.00048096378800580156",
            "extra": "mean: 41.924376666666895 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 861.2914572796955,
            "unit": "iter/sec",
            "range": "stddev: 0.00001221259980312791",
            "extra": "mean: 1.161047159527626 msec\nrounds: 677"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 6049.388309973868,
            "unit": "iter/sec",
            "range": "stddev: 0.00001713658826859827",
            "extra": "mean: 165.3059695889021 usec\nrounds: 4768"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 33896.45945729332,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032779786076122826",
            "extra": "mean: 29.501606244744103 usec\nrounds: 7078"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1780.8232169226453,
            "unit": "iter/sec",
            "range": "stddev: 0.000021655407882337523",
            "extra": "mean: 561.5380518949273 usec\nrounds: 1214"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1572.0133881056258,
            "unit": "iter/sec",
            "range": "stddev: 0.000017540669822023874",
            "extra": "mean: 636.1268978790711 usec\nrounds: 1273"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3203.916810239077,
            "unit": "iter/sec",
            "range": "stddev: 0.000015905144478266645",
            "extra": "mean: 312.11796661018167 usec\nrounds: 2366"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2110.152017692445,
            "unit": "iter/sec",
            "range": "stddev: 0.000018761015458659095",
            "extra": "mean: 473.899506583203 usec\nrounds: 1595"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 7513.3864507809485,
            "unit": "iter/sec",
            "range": "stddev: 0.000008088619852621714",
            "extra": "mean: 133.09577599273615 usec\nrounds: 4357"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 105.58982583841033,
            "unit": "iter/sec",
            "range": "stddev: 0.000534664525964708",
            "extra": "mean: 9.470609427184325 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 8006.792293880103,
            "unit": "iter/sec",
            "range": "stddev: 0.000007175580998875992",
            "extra": "mean: 124.89396043960554 usec\nrounds: 3185"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 12353.394979325658,
            "unit": "iter/sec",
            "range": "stddev: 0.000004922689680767603",
            "extra": "mean: 80.94940716083116 usec\nrounds: 9915"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 38120.8216186438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022374158178241445",
            "extra": "mean: 26.232383184283957 usec\nrounds: 5697"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 11729.744365821558,
            "unit": "iter/sec",
            "range": "stddev: 0.000006813881195772315",
            "extra": "mean: 85.25334984399376 usec\nrounds: 9933"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 13438.430733811021,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032292219180249773",
            "extra": "mean: 74.4134504845127 usec\nrounds: 12693"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 4918.7726742505165,
            "unit": "iter/sec",
            "range": "stddev: 0.000006909574392278093",
            "extra": "mean: 203.30274770268218 usec\nrounds: 3591"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 83289.0600418344,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141359756456152",
            "extra": "mean: 12.006378742871158 usec\nrounds: 14461"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 339634.31740517897,
            "unit": "iter/sec",
            "range": "stddev: 5.265636832736474e-7",
            "extra": "mean: 2.944343220791243 usec\nrounds: 80135"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1342.6984475205113,
            "unit": "iter/sec",
            "range": "stddev: 0.000015887599244462925",
            "extra": "mean: 744.7688658958726 usec\nrounds: 865"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 192.85276550255148,
            "unit": "iter/sec",
            "range": "stddev: 0.003499031139370338",
            "extra": "mean: 5.185302878048538 msec\nrounds: 205"
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
          "id": "060ba9f2f89fc54572f27067582da36d3f2c26c3",
          "message": "Merge pull request #49 from Roy-Kid/docs/drop-zh-tree\n\ndocs: drop the Chinese docs tree; remove the wall-clock scaling test",
          "timestamp": "2026-07-24T16:42:12+02:00",
          "tree_id": "0519158ea9ce95b1596ae026882396bf2f4ac968",
          "url": "https://github.com/MolCrafts/molpy/commit/060ba9f2f89fc54572f27067582da36d3f2c26c3"
        },
        "date": 1784904196736,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7718.028173813601,
            "unit": "iter/sec",
            "range": "stddev: 0.000004951684266313497",
            "extra": "mean: 129.56677242937351 usec\nrounds: 4882"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113721.01717808041,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010998231999278602",
            "extra": "mean: 8.793449309673857 usec\nrounds: 38390"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48020.26941268689,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016926961799630059",
            "extra": "mean: 20.824539558618163 usec\nrounds: 19351"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.854521194820926,
            "unit": "iter/sec",
            "range": "stddev: 0.0011925882583407117",
            "extra": "mean: 16.432632783332224 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 31299.920067635627,
            "unit": "iter/sec",
            "range": "stddev: 0.000009492642056279198",
            "extra": "mean: 31.94896337879176 usec\nrounds: 9885"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18205.68937569555,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035284330004333626",
            "extra": "mean: 54.92788432032637 usec\nrounds: 11869"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 363.1426812305873,
            "unit": "iter/sec",
            "range": "stddev: 0.000020452405181945948",
            "extra": "mean: 2.7537385487469677 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3541.19029303303,
            "unit": "iter/sec",
            "range": "stddev: 0.000022715839744569866",
            "extra": "mean: 282.3909243079677 usec\nrounds: 2312"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 472256.6537808223,
            "unit": "iter/sec",
            "range": "stddev: 5.36024406146355e-7",
            "extra": "mean: 2.1174926641988767 usec\nrounds: 68227"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 806.7856448128794,
            "unit": "iter/sec",
            "range": "stddev: 0.00003290488111233869",
            "extra": "mean: 1.2394866051836277 msec\nrounds: 656"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52355.45570974229,
            "unit": "iter/sec",
            "range": "stddev: 0.000002829940057567917",
            "extra": "mean: 19.10020620475509 usec\nrounds: 11862"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 439.03036105697686,
            "unit": "iter/sec",
            "range": "stddev: 0.000047343445676320905",
            "extra": "mean: 2.2777467999991488 msec\nrounds: 395"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 214002.15646271943,
            "unit": "iter/sec",
            "range": "stddev: 8.152102220262735e-7",
            "extra": "mean: 4.672850108284804 usec\nrounds: 50770"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 122437.06369281575,
            "unit": "iter/sec",
            "range": "stddev: 0.000001096883935031355",
            "extra": "mean: 8.167461468276596 usec\nrounds: 36204"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 148333.04510352557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013987378603325885",
            "extra": "mean: 6.741586133434215 usec\nrounds: 44164"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 74203.46266509051,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014459008340712972",
            "extra": "mean: 13.47646004760444 usec\nrounds: 27295"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3554.596709674191,
            "unit": "iter/sec",
            "range": "stddev: 0.0000068147119461434815",
            "extra": "mean: 281.3258666667866 usec\nrounds: 2880"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15707.805741544125,
            "unit": "iter/sec",
            "range": "stddev: 0.000003546112082161375",
            "extra": "mean: 63.662615673632395 usec\nrounds: 11714"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16735.542907270345,
            "unit": "iter/sec",
            "range": "stddev: 0.000004390880884782988",
            "extra": "mean: 59.753066006933935 usec\nrounds: 12832"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.6255339303041,
            "unit": "iter/sec",
            "range": "stddev: 0.00022389083654186212",
            "extra": "mean: 12.718514584415137 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1268.7834836510206,
            "unit": "iter/sec",
            "range": "stddev: 0.000008616662206185019",
            "extra": "mean: 788.1565396188987 usec\nrounds: 997"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 138.08780783366905,
            "unit": "iter/sec",
            "range": "stddev: 0.000048993523763916275",
            "extra": "mean: 7.241768956203074 msec\nrounds: 137"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 45193.73089274247,
            "unit": "iter/sec",
            "range": "stddev: 0.000001848191277244846",
            "extra": "mean: 22.12696275006114 usec\nrounds: 25933"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1079.2245552809054,
            "unit": "iter/sec",
            "range": "stddev: 0.00008763610174338573",
            "extra": "mean: 926.5912224724311 usec\nrounds: 890"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1246.2202534595128,
            "unit": "iter/sec",
            "range": "stddev: 0.000010913442766454247",
            "extra": "mean: 802.4263746508658 usec\nrounds: 1073"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 109204.76007352784,
            "unit": "iter/sec",
            "range": "stddev: 0.000001015263307491577",
            "extra": "mean: 9.157109995266666 usec\nrounds: 54366"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 91554.11925086247,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016215299100785055",
            "extra": "mean: 10.922501447039803 usec\nrounds: 40427"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 77802.62901551205,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013784048191137915",
            "extra": "mean: 12.853036107566792 usec\nrounds: 49685"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 102424.1833719596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012277758287119586",
            "extra": "mean: 9.763319238469684 usec\nrounds: 58201"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40072.06927876987,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020744110422398785",
            "extra": "mean: 24.955037710762756 usec\nrounds: 16653"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 178687.22146451144,
            "unit": "iter/sec",
            "range": "stddev: 8.690686002733919e-7",
            "extra": "mean: 5.596371088005346 usec\nrounds: 28023"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 177981.15968210206,
            "unit": "iter/sec",
            "range": "stddev: 8.977098381217091e-7",
            "extra": "mean: 5.618572222959624 usec\nrounds: 35072"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 179492.71138106266,
            "unit": "iter/sec",
            "range": "stddev: 8.781897651250548e-7",
            "extra": "mean: 5.571256862218778 usec\nrounds: 44701"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 162456.3367961658,
            "unit": "iter/sec",
            "range": "stddev: 9.54035272467119e-7",
            "extra": "mean: 6.1555001160385725 usec\nrounds: 38793"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 167255.2658047631,
            "unit": "iter/sec",
            "range": "stddev: 9.068848812900674e-7",
            "extra": "mean: 5.978885000651034 usec\nrounds: 43435"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180446.99601001779,
            "unit": "iter/sec",
            "range": "stddev: 8.767578088185318e-7",
            "extra": "mean: 5.541793557729737 usec\nrounds: 41973"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.389861727000106,
            "unit": "iter/sec",
            "range": "stddev: 0.00025872716406533065",
            "extra": "mean: 42.75356612500403 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 854.9568944227658,
            "unit": "iter/sec",
            "range": "stddev: 0.000012912727311063319",
            "extra": "mean: 1.1696496121891171 msec\nrounds: 722"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4498.945222738702,
            "unit": "iter/sec",
            "range": "stddev: 0.0000066587494135220195",
            "extra": "mean: 222.27432220018383 usec\nrounds: 3563"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34676.98601905416,
            "unit": "iter/sec",
            "range": "stddev: 0.000003903293575087686",
            "extra": "mean: 28.837569662211255 usec\nrounds: 6955"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1447.1855053105842,
            "unit": "iter/sec",
            "range": "stddev: 0.000021004138916705004",
            "extra": "mean: 690.9964177573679 usec\nrounds: 1070"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1288.2910201797547,
            "unit": "iter/sec",
            "range": "stddev: 0.000024142760017885378",
            "extra": "mean: 776.2221301988664 usec\nrounds: 1106"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2555.206289961838,
            "unit": "iter/sec",
            "range": "stddev: 0.000020834615404333078",
            "extra": "mean: 391.35783436684284 usec\nrounds: 2095"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1468.4015270688153,
            "unit": "iter/sec",
            "range": "stddev: 0.000022028050828079016",
            "extra": "mean: 681.0126396396316 usec\nrounds: 1221"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5360.282238373807,
            "unit": "iter/sec",
            "range": "stddev: 0.000036619987799081975",
            "extra": "mean: 186.55734073871795 usec\nrounds: 4006"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 110.97470397881484,
            "unit": "iter/sec",
            "range": "stddev: 0.00035783062321958847",
            "extra": "mean: 9.01106255882332 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5918.068099065004,
            "unit": "iter/sec",
            "range": "stddev: 0.000007426620964860289",
            "extra": "mean: 168.974061004467 usec\nrounds: 2967"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10927.768922064375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000040866519004435215",
            "extra": "mean: 91.5099877323439 usec\nrounds: 8233"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 28900.619838338196,
            "unit": "iter/sec",
            "range": "stddev: 0.000005108659735560294",
            "extra": "mean: 34.601334005765764 usec\nrounds: 6940"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13695.89192571026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034884508199489697",
            "extra": "mean: 73.01459484524523 usec\nrounds: 12183"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14290.105084372235,
            "unit": "iter/sec",
            "range": "stddev: 0.000004768372038780764",
            "extra": "mean: 69.97849169727992 usec\nrounds: 13429"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5194.308741323257,
            "unit": "iter/sec",
            "range": "stddev: 0.000019280579686753127",
            "extra": "mean: 192.51839846263135 usec\nrounds: 4033"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 79052.70969790603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019823285860438846",
            "extra": "mean: 12.649787765927627 usec\nrounds: 11280"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 331644.1466722214,
            "unit": "iter/sec",
            "range": "stddev: 8.556392374975442e-7",
            "extra": "mean: 3.0152801128383677 usec\nrounds: 82556"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1401.9537107672543,
            "unit": "iter/sec",
            "range": "stddev: 0.000016365814392575895",
            "extra": "mean: 713.2903121692407 usec\nrounds: 945"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 189.87439385939606,
            "unit": "iter/sec",
            "range": "stddev: 0.0032312978960938234",
            "extra": "mean: 5.266639590910349 msec\nrounds: 198"
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
          "id": "f47de7cf110fa1bb2dc60269690dd63e9ae31ede",
          "message": "Merge pull request #51 from Roy-Kid/release/v0.10.0\n\nrelease: v0.10.0",
          "timestamp": "2026-07-29T17:01:11+02:00",
          "tree_id": "ea73d4b82d01b28b23e13a4d1ba072da5446d97e",
          "url": "https://github.com/MolCrafts/molpy/commit/f47de7cf110fa1bb2dc60269690dd63e9ae31ede"
        },
        "date": 1785337344792,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7543.647414311166,
            "unit": "iter/sec",
            "range": "stddev: 0.000006776716633676045",
            "extra": "mean: 132.56186895783134 usec\nrounds: 4510"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113110.49547667376,
            "unit": "iter/sec",
            "range": "stddev: 0.000001074860589154496",
            "extra": "mean: 8.840912558872358 usec\nrounds: 34183"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48253.649765833405,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017954953870106223",
            "extra": "mean: 20.723820992874664 usec\nrounds: 20083"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 52.96588948022934,
            "unit": "iter/sec",
            "range": "stddev: 0.002177135832519354",
            "extra": "mean: 18.880075645161618 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 30309.573204529617,
            "unit": "iter/sec",
            "range": "stddev: 0.00006361845614451361",
            "extra": "mean: 32.9928763183823 usec\nrounds: 12136"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18236.781002387277,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033619331793859803",
            "extra": "mean: 54.83423855718262 usec\nrounds: 10618"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 363.50867295342545,
            "unit": "iter/sec",
            "range": "stddev: 0.00003123526858854154",
            "extra": "mean: 2.7509660000000196 msec\nrounds: 351"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3481.465760144307,
            "unit": "iter/sec",
            "range": "stddev: 0.00003991375446434259",
            "extra": "mean: 287.2353396227427 usec\nrounds: 2226"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 467999.79038555484,
            "unit": "iter/sec",
            "range": "stddev: 5.335993330026583e-7",
            "extra": "mean: 2.13675309379127 usec\nrounds: 66989"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 807.3508579874964,
            "unit": "iter/sec",
            "range": "stddev: 0.000037725997988923024",
            "extra": "mean: 1.2386188608168758 msec\nrounds: 661"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 51685.771041668115,
            "unit": "iter/sec",
            "range": "stddev: 0.000002662159512989032",
            "extra": "mean: 19.34768466922586 usec\nrounds: 12276"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 449.288731942162,
            "unit": "iter/sec",
            "range": "stddev: 0.00005032568915606946",
            "extra": "mean: 2.2257402176040606 msec\nrounds: 409"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 215784.86518381888,
            "unit": "iter/sec",
            "range": "stddev: 7.209742844311105e-7",
            "extra": "mean: 4.634245312562298 usec\nrounds: 40907"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 122809.81485150193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010072235019606636",
            "extra": "mean: 8.142671668458837 usec\nrounds: 30789"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 150779.5448512177,
            "unit": "iter/sec",
            "range": "stddev: 9.170420182355297e-7",
            "extra": "mean: 6.632199354274174 usec\nrounds: 40265"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 74892.93125968862,
            "unit": "iter/sec",
            "range": "stddev: 0.000001299247323267039",
            "extra": "mean: 13.352394988153623 usec\nrounds: 29849"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3662.8225550422344,
            "unit": "iter/sec",
            "range": "stddev: 0.000016448142759104802",
            "extra": "mean: 273.0134984626547 usec\nrounds: 2927"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15201.08630776695,
            "unit": "iter/sec",
            "range": "stddev: 0.000004396292978213423",
            "extra": "mean: 65.78477220335583 usec\nrounds: 10843"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17265.52891092319,
            "unit": "iter/sec",
            "range": "stddev: 0.000010676258316012412",
            "extra": "mean: 57.91887437443872 usec\nrounds: 13389"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.80931392258555,
            "unit": "iter/sec",
            "range": "stddev: 0.00006211424356390171",
            "extra": "mean: 12.688855545453686 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1259.7068584605904,
            "unit": "iter/sec",
            "range": "stddev: 0.000010314493324532566",
            "extra": "mean: 793.8354810753655 usec\nrounds: 1004"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 135.96952882068163,
            "unit": "iter/sec",
            "range": "stddev: 0.0005857270831049349",
            "extra": "mean: 7.354588992647117 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44849.610415045245,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017529852330700321",
            "extra": "mean: 22.29673771401457 usec\nrounds: 23360"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1146.318708703632,
            "unit": "iter/sec",
            "range": "stddev: 0.000060656448411143195",
            "extra": "mean: 872.3577417059665 usec\nrounds: 844"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1288.7580179993372,
            "unit": "iter/sec",
            "range": "stddev: 0.000014971479959616265",
            "extra": "mean: 775.9408562612832 usec\nrounds: 1134"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 108941.35177602585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011618372491621595",
            "extra": "mean: 9.17925088772457 usec\nrounds: 54072"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 94819.03051046917,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011448395153710937",
            "extra": "mean: 10.546406081314952 usec\nrounds: 47062"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 78635.97727307922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020146797753024547",
            "extra": "mean: 12.716825487235942 usec\nrounds: 50386"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 103914.72103775959,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013936160852537595",
            "extra": "mean: 9.62327560535556 usec\nrounds: 59099"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39790.51586932257,
            "unit": "iter/sec",
            "range": "stddev: 0.000005487099682679741",
            "extra": "mean: 25.131616872828065 usec\nrounds: 16334"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 179392.78461872286,
            "unit": "iter/sec",
            "range": "stddev: 9.302766724426949e-7",
            "extra": "mean: 5.574360206991469 usec\nrounds: 29186"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 180052.69259947145,
            "unit": "iter/sec",
            "range": "stddev: 8.929034820637891e-7",
            "extra": "mean: 5.553929716699697 usec\nrounds: 46839"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 177977.93815913296,
            "unit": "iter/sec",
            "range": "stddev: 9.39083573751335e-7",
            "extra": "mean: 5.618673922977374 usec\nrounds: 45891"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 163071.63071945112,
            "unit": "iter/sec",
            "range": "stddev: 9.079725407302145e-7",
            "extra": "mean: 6.132274483232481 usec\nrounds: 41314"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 169313.2254817898,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010350775728118915",
            "extra": "mean: 5.906213157031571 usec\nrounds: 36832"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 179205.14447359846,
            "unit": "iter/sec",
            "range": "stddev: 9.896871923796244e-7",
            "extra": "mean: 5.580196946563249 usec\nrounds: 44540"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.30324923020055,
            "unit": "iter/sec",
            "range": "stddev: 0.000982258572726392",
            "extra": "mean: 42.91247070833452 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 853.8289482627329,
            "unit": "iter/sec",
            "range": "stddev: 0.00001536355238689885",
            "extra": "mean: 1.1711947715460787 msec\nrounds: 731"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 5226.766817788843,
            "unit": "iter/sec",
            "range": "stddev: 0.000006875144744619035",
            "extra": "mean: 191.32286456640608 usec\nrounds: 4253"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34916.078842918985,
            "unit": "iter/sec",
            "range": "stddev: 0.000007662435891980846",
            "extra": "mean: 28.64010029587847 usec\nrounds: 6421"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1424.258484240059,
            "unit": "iter/sec",
            "range": "stddev: 0.000024511429230466955",
            "extra": "mean: 702.1197423539095 usec\nrounds: 1079"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1273.2746665603372,
            "unit": "iter/sec",
            "range": "stddev: 0.00002488834226926627",
            "extra": "mean: 785.3764990875303 usec\nrounds: 1096"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2569.9142299385767,
            "unit": "iter/sec",
            "range": "stddev: 0.000019038798823599218",
            "extra": "mean: 389.1180446220187 usec\nrounds: 2129"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1437.5146558839551,
            "unit": "iter/sec",
            "range": "stddev: 0.00007486195852735341",
            "extra": "mean: 695.6450815348946 usec\nrounds: 1251"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5379.073067999661,
            "unit": "iter/sec",
            "range": "stddev: 0.00003202449949373217",
            "extra": "mean: 185.90563603774848 usec\nrounds: 3907"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 111.51022294852088,
            "unit": "iter/sec",
            "range": "stddev: 0.00012829376390101043",
            "extra": "mean: 8.967787648148223 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 6016.4776222306855,
            "unit": "iter/sec",
            "range": "stddev: 0.000007479010084162629",
            "extra": "mean: 166.2102084955877 usec\nrounds: 3343"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11115.858105845873,
            "unit": "iter/sec",
            "range": "stddev: 0.000010331985013147928",
            "extra": "mean: 89.96156576288934 usec\nrounds: 9405"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29830.70119174701,
            "unit": "iter/sec",
            "range": "stddev: 0.000004933731621690142",
            "extra": "mean: 33.5225107037263 usec\nrounds: 6820"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13958.832052658076,
            "unit": "iter/sec",
            "range": "stddev: 0.00000500730124233389",
            "extra": "mean: 71.63923143624166 usec\nrounds: 12457"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14588.869388695068,
            "unit": "iter/sec",
            "range": "stddev: 0.000003838655000594401",
            "extra": "mean: 68.5454076910786 usec\nrounds: 13704"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5357.956452059329,
            "unit": "iter/sec",
            "range": "stddev: 0.000008676560638940045",
            "extra": "mean: 186.63832170857793 usec\nrounds: 3699"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 77750.82455078934,
            "unit": "iter/sec",
            "range": "stddev: 0.000002199822273093042",
            "extra": "mean: 12.861599935146254 usec\nrounds: 15425"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 326714.1911693957,
            "unit": "iter/sec",
            "range": "stddev: 6.734242130034327e-7",
            "extra": "mean: 3.0607791979305765 usec\nrounds: 74483"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1428.0853237660317,
            "unit": "iter/sec",
            "range": "stddev: 0.000013920464004350768",
            "extra": "mean: 700.2382724324066 usec\nrounds: 925"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 187.55070653582194,
            "unit": "iter/sec",
            "range": "stddev: 0.0028530674509784394",
            "extra": "mean: 5.331891404040119 msec\nrounds: 198"
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
          "id": "29b0090dbd4e62d99b5ea9196dd54c0c51cba09b",
          "message": "Merge pull request #52 from Roy-Kid/master\n\nchore: minor-line molrs pin, drop CHANGELOG (no release)",
          "timestamp": "2026-07-30T17:05:51+02:00",
          "tree_id": "4640ab2edfbfabd5d8b5752791a1402e2688d52a",
          "url": "https://github.com/MolCrafts/molpy/commit/29b0090dbd4e62d99b5ea9196dd54c0c51cba09b"
        },
        "date": 1785424014032,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 11274.120835598951,
            "unit": "iter/sec",
            "range": "stddev: 0.000004132830995909617",
            "extra": "mean: 88.69871226166201 usec\nrounds: 5717"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113784.61060340036,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013035013434612596",
            "extra": "mean: 8.788534712181155 usec\nrounds: 39453"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48456.39878676862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016190300813775662",
            "extra": "mean: 20.637109340305688 usec\nrounds: 18008"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.406858214702424,
            "unit": "iter/sec",
            "range": "stddev: 0.0012576354859346061",
            "extra": "mean: 16.554411693548563 msec\nrounds: 62"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 35937.15264490834,
            "unit": "iter/sec",
            "range": "stddev: 0.000005508249078365516",
            "extra": "mean: 27.826355913082683 usec\nrounds: 9758"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18099.96577791624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032860069396555013",
            "extra": "mean: 55.24872324455439 usec\nrounds: 10923"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 390.5146161009517,
            "unit": "iter/sec",
            "range": "stddev: 0.000019699471687834915",
            "extra": "mean: 2.560723616402339 msec\nrounds: 378"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3896.2906329855373,
            "unit": "iter/sec",
            "range": "stddev: 0.000027171566937194972",
            "extra": "mean: 256.65436544546185 usec\nrounds: 2211"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 462226.14283911383,
            "unit": "iter/sec",
            "range": "stddev: 6.646872612077724e-7",
            "extra": "mean: 2.163443187911741 usec\nrounds: 65347"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1056.1699437942755,
            "unit": "iter/sec",
            "range": "stddev: 0.000027210369991340746",
            "extra": "mean: 946.8173241206945 usec\nrounds: 796"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 58607.36840585707,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020678699972236845",
            "extra": "mean: 17.062700940178413 usec\nrounds: 11700"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 559.9316319264003,
            "unit": "iter/sec",
            "range": "stddev: 0.00003023487709136945",
            "extra": "mean: 1.7859323227722987 msec\nrounds: 505"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 218692.14142959513,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010804720694968563",
            "extra": "mean: 4.57263801736532 usec\nrounds: 21769"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 117646.59603709869,
            "unit": "iter/sec",
            "range": "stddev: 9.060265541106486e-7",
            "extra": "mean: 8.500033436451147 usec\nrounds: 27156"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 145596.43361187316,
            "unit": "iter/sec",
            "range": "stddev: 9.437867906305891e-7",
            "extra": "mean: 6.868300103186399 usec\nrounds: 39733"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 70697.7593044973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013856580727463068",
            "extra": "mean: 14.144719858701194 usec\nrounds: 25480"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 4031.0820668112265,
            "unit": "iter/sec",
            "range": "stddev: 0.000008051477550707166",
            "extra": "mean: 248.0723496634358 usec\nrounds: 2674"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15345.731873702245,
            "unit": "iter/sec",
            "range": "stddev: 0.000003474700692261038",
            "extra": "mean: 65.16469909875626 usec\nrounds: 10874"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16752.62084946481,
            "unit": "iter/sec",
            "range": "stddev: 0.000003155594985665898",
            "extra": "mean: 59.69215258828869 usec\nrounds: 10335"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.65323961420002,
            "unit": "iter/sec",
            "range": "stddev: 0.00012475660309171658",
            "extra": "mean: 12.554417181818266 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1141.901275553549,
            "unit": "iter/sec",
            "range": "stddev: 0.00000877661270167947",
            "extra": "mean: 875.7324485124506 usec\nrounds: 874"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 140.34060954935757,
            "unit": "iter/sec",
            "range": "stddev: 0.00032816058034141086",
            "extra": "mean: 7.125521281481263 msec\nrounds: 135"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 41708.349714478434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018126589311014974",
            "extra": "mean: 23.976014559330906 usec\nrounds: 21773"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1119.8212983633696,
            "unit": "iter/sec",
            "range": "stddev: 0.00010367193005446936",
            "extra": "mean: 892.9996254415863 usec\nrounds: 849"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1095.1119001394193,
            "unit": "iter/sec",
            "range": "stddev: 0.000011239735876043063",
            "extra": "mean: 913.1486927250901 usec\nrounds: 921"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 108643.9617236,
            "unit": "iter/sec",
            "range": "stddev: 0.000001081100708406059",
            "extra": "mean: 9.204377161282924 usec\nrounds: 51763"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 97183.22954000968,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010275991375981936",
            "extra": "mean: 10.289841207513142 usec\nrounds: 51180"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 83520.60356989803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012775371916908991",
            "extra": "mean: 11.973093551258934 usec\nrounds: 51576"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 109547.87947940669,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010839486410017852",
            "extra": "mean: 9.12842863551717 usec\nrounds: 60191"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 39426.31322657483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018319700101761774",
            "extra": "mean: 25.363771505928234 usec\nrounds: 14775"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 172032.75555512757,
            "unit": "iter/sec",
            "range": "stddev: 8.317319230393473e-7",
            "extra": "mean: 5.812846494105896 usec\nrounds: 21478"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 171954.02937821523,
            "unit": "iter/sec",
            "range": "stddev: 8.56379437934939e-7",
            "extra": "mean: 5.8155078052895535 usec\nrounds: 41126"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 172720.90249265332,
            "unit": "iter/sec",
            "range": "stddev: 8.729120569770217e-7",
            "extra": "mean: 5.789687209644676 usec\nrounds: 39608"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 161667.75596559834,
            "unit": "iter/sec",
            "range": "stddev: 8.032700977598637e-7",
            "extra": "mean: 6.185525332663072 usec\nrounds: 35547"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 167905.28584962134,
            "unit": "iter/sec",
            "range": "stddev: 7.89898636145829e-7",
            "extra": "mean: 5.955738647177648 usec\nrounds: 30499"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180539.81719125898,
            "unit": "iter/sec",
            "range": "stddev: 7.552519117923412e-7",
            "extra": "mean: 5.538944347886578 usec\nrounds: 36818"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.693121579173706,
            "unit": "iter/sec",
            "range": "stddev: 0.0004883119363435463",
            "extra": "mean: 42.20634232000066 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 854.0591613534538,
            "unit": "iter/sec",
            "range": "stddev: 0.000032554848090696785",
            "extra": "mean: 1.170879074015516 msec\nrounds: 635"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 6912.434009855927,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051544499512648255",
            "extra": "mean: 144.66684218238817 usec\nrounds: 6140"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 33147.52367432717,
            "unit": "iter/sec",
            "range": "stddev: 0.00002527104156879842",
            "extra": "mean: 30.168166099674657 usec\nrounds: 6761"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1765.240115871199,
            "unit": "iter/sec",
            "range": "stddev: 0.000018378506215108496",
            "extra": "mean: 566.4951702655308 usec\nrounds: 1204"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1585.6819781969104,
            "unit": "iter/sec",
            "range": "stddev: 0.00002544530579093477",
            "extra": "mean: 630.643479430287 usec\nrounds: 1264"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3236.9351897242186,
            "unit": "iter/sec",
            "range": "stddev: 0.000013130284304371911",
            "extra": "mean: 308.9342051625069 usec\nrounds: 2247"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2122.3325278911893,
            "unit": "iter/sec",
            "range": "stddev: 0.00002113078701901199",
            "extra": "mean: 471.1796982132808 usec\nrounds: 1511"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 7593.780516874867,
            "unit": "iter/sec",
            "range": "stddev: 0.000007802384569164861",
            "extra": "mean: 131.68671359118218 usec\nrounds: 4319"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 106.28103773253657,
            "unit": "iter/sec",
            "range": "stddev: 0.00013427447616017515",
            "extra": "mean: 9.409016145632371 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 7690.662813047508,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071427049304795826",
            "extra": "mean: 130.0278044050327 usec\nrounds: 3042"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 12219.00948413228,
            "unit": "iter/sec",
            "range": "stddev: 0.000009501126142914634",
            "extra": "mean: 81.83969423205778 usec\nrounds: 10194"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 37703.94130623607,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025892423202715667",
            "extra": "mean: 26.522426180273207 usec\nrounds: 6333"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 12031.685737169739,
            "unit": "iter/sec",
            "range": "stddev: 0.000005484709265385806",
            "extra": "mean: 83.11387297215377 usec\nrounds: 10171"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 13471.805376772925,
            "unit": "iter/sec",
            "range": "stddev: 0.000003300540010958681",
            "extra": "mean: 74.22910085415313 usec\nrounds: 12761"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 4975.688571011717,
            "unit": "iter/sec",
            "range": "stddev: 0.000010473377643772088",
            "extra": "mean: 200.9772086271605 usec\nrounds: 3686"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 84080.26888380505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014023083269079234",
            "extra": "mean: 11.893396789464989 usec\nrounds: 11462"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 338307.22262332804,
            "unit": "iter/sec",
            "range": "stddev: 5.500823760738665e-7",
            "extra": "mean: 2.955893144242451 usec\nrounds: 86453"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1336.7641219775032,
            "unit": "iter/sec",
            "range": "stddev: 0.00001541574527747886",
            "extra": "mean: 748.0751342433391 usec\nrounds: 879"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 195.97870669873075,
            "unit": "iter/sec",
            "range": "stddev: 0.003036501595566989",
            "extra": "mean: 5.102595158652899 msec\nrounds: 208"
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
          "id": "6637333159cdddce200e12324676aaadb753e627",
          "message": "Merge pull request #53 from Roy-Kid/ci/bump-actions\n\nci: bump GitHub Actions to Node-24-ready majors",
          "timestamp": "2026-07-31T11:48:19+02:00",
          "tree_id": "1ce43e457ddc31b893099782b7279ea8a8019800",
          "url": "https://github.com/MolCrafts/molpy/commit/6637333159cdddce200e12324676aaadb753e627"
        },
        "date": 1785491361466,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 14218.178735469957,
            "unit": "iter/sec",
            "range": "stddev: 0.000007269812905212768",
            "extra": "mean: 70.33249606753847 usec\nrounds: 5213"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 138147.30686072892,
            "unit": "iter/sec",
            "range": "stddev: 9.226690994023176e-7",
            "extra": "mean: 7.238649979678103 usec\nrounds: 41829"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 63815.269017583094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014193085158484418",
            "extra": "mean: 15.670230892930482 usec\nrounds: 19901"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 81.91364925426352,
            "unit": "iter/sec",
            "range": "stddev: 0.0009495633539384849",
            "extra": "mean: 12.207977658228321 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 35158.917431549715,
            "unit": "iter/sec",
            "range": "stddev: 0.000007071687409379875",
            "extra": "mean: 28.442286425538633 usec\nrounds: 12855"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 30385.190901905728,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020920323720840178",
            "extra": "mean: 32.91076903970615 usec\nrounds: 11673"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 442.10230848961754,
            "unit": "iter/sec",
            "range": "stddev: 0.00007972916907461017",
            "extra": "mean: 2.2619198787184898 msec\nrounds: 437"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 5908.769379562345,
            "unit": "iter/sec",
            "range": "stddev: 0.00000872063390203456",
            "extra": "mean: 169.23997803313634 usec\nrounds: 2959"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 695415.4197009411,
            "unit": "iter/sec",
            "range": "stddev: 3.569967384357864e-7",
            "extra": "mean: 1.4379893969421091 usec\nrounds: 76676"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1552.7612941959078,
            "unit": "iter/sec",
            "range": "stddev: 0.000044055299520828364",
            "extra": "mean: 644.0139921943679 usec\nrounds: 1153"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 83730.79909105544,
            "unit": "iter/sec",
            "range": "stddev: 0.000001201397427288654",
            "extra": "mean: 11.943036622790636 usec\nrounds: 15646"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 831.8791028807358,
            "unit": "iter/sec",
            "range": "stddev: 0.000054138181261380426",
            "extra": "mean: 1.2020977525905796 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 277477.7537926129,
            "unit": "iter/sec",
            "range": "stddev: 6.40026064299691e-7",
            "extra": "mean: 3.6038925151001506 usec\nrounds: 36126"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 144228.68029740467,
            "unit": "iter/sec",
            "range": "stddev: 5.949990491487943e-7",
            "extra": "mean: 6.933433752135598 usec\nrounds: 32235"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 182057.69664868093,
            "unit": "iter/sec",
            "range": "stddev: 6.530227360329299e-7",
            "extra": "mean: 5.492764208314207 usec\nrounds: 44217"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 84439.4636138779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015903685286048273",
            "extra": "mean: 11.842803793410724 usec\nrounds: 23567"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 4764.293707338169,
            "unit": "iter/sec",
            "range": "stddev: 0.000014619064747030567",
            "extra": "mean: 209.89470033297007 usec\nrounds: 3604"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 17405.25930036938,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041908600593288496",
            "extra": "mean: 57.453898430503564 usec\nrounds: 11214"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16463.69874239614,
            "unit": "iter/sec",
            "range": "stddev: 0.000003812854160820976",
            "extra": "mean: 60.739692559173925 usec\nrounds: 13762"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 113.28620090467857,
            "unit": "iter/sec",
            "range": "stddev: 0.0002163780185061163",
            "extra": "mean: 8.827200418181746 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1504.1504146198802,
            "unit": "iter/sec",
            "range": "stddev: 0.00003180278006532854",
            "extra": "mean: 664.8271278459302 usec\nrounds: 1142"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 198.89779825941721,
            "unit": "iter/sec",
            "range": "stddev: 0.00014612918482677324",
            "extra": "mean: 5.027707741117003 msec\nrounds: 197"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 58822.74989570214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014577322846249786",
            "extra": "mean: 17.000225283127484 usec\nrounds: 23486"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1344.5330645794572,
            "unit": "iter/sec",
            "range": "stddev: 0.00008356138550323012",
            "extra": "mean: 743.7526278409374 usec\nrounds: 1056"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1918.8144979566991,
            "unit": "iter/sec",
            "range": "stddev: 0.00001533704762925335",
            "extra": "mean: 521.15512003108 usec\nrounds: 1283"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 126072.73330667612,
            "unit": "iter/sec",
            "range": "stddev: 8.095883185809466e-7",
            "extra": "mean: 7.931929242522781 usec\nrounds: 45663"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 126731.83159110066,
            "unit": "iter/sec",
            "range": "stddev: 8.239631106207376e-7",
            "extra": "mean: 7.890677404761991 usec\nrounds: 52989"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 112961.58997936238,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011453513904774303",
            "extra": "mean: 8.852566612976108 usec\nrounds: 52152"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 138795.35401597343,
            "unit": "iter/sec",
            "range": "stddev: 8.056784985763807e-7",
            "extra": "mean: 7.204852115474369 usec\nrounds: 57741"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 54173.71975058538,
            "unit": "iter/sec",
            "range": "stddev: 0.000001791400153328554",
            "extra": "mean: 18.4591348831865 usec\nrounds: 17808"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 224481.3353332427,
            "unit": "iter/sec",
            "range": "stddev: 5.944498671862433e-7",
            "extra": "mean: 4.454713344053747 usec\nrounds: 24880"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 231766.082602843,
            "unit": "iter/sec",
            "range": "stddev: 6.900976833113778e-7",
            "extra": "mean: 4.314695182183371 usec\nrounds: 37465"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 231268.6010230114,
            "unit": "iter/sec",
            "range": "stddev: 5.703513546637115e-7",
            "extra": "mean: 4.323976517246711 usec\nrounds: 40583"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 215824.94242923163,
            "unit": "iter/sec",
            "range": "stddev: 7.094145673913395e-7",
            "extra": "mean: 4.633384764265127 usec\nrounds: 37714"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 219714.45993346663,
            "unit": "iter/sec",
            "range": "stddev: 5.708603610844567e-7",
            "extra": "mean: 4.551361800688118 usec\nrounds: 33676"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 237397.0736720075,
            "unit": "iter/sec",
            "range": "stddev: 5.583959356242605e-7",
            "extra": "mean: 4.212351839608688 usec\nrounds: 43787"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 32.514298226688446,
            "unit": "iter/sec",
            "range": "stddev: 0.0008062463263440348",
            "extra": "mean: 30.755699939393995 msec\nrounds: 33"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 1083.2918790142226,
            "unit": "iter/sec",
            "range": "stddev: 0.000023738739157287967",
            "extra": "mean: 923.1122464519749 usec\nrounds: 775"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 8496.425595865727,
            "unit": "iter/sec",
            "range": "stddev: 0.000003714919735837826",
            "extra": "mean: 117.6965523580398 usec\nrounds: 6255"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 26927.371432568332,
            "unit": "iter/sec",
            "range": "stddev: 0.000005957796629393722",
            "extra": "mean: 37.136933417515536 usec\nrounds: 8711"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 2475.3283758831485,
            "unit": "iter/sec",
            "range": "stddev: 0.000027300240818463603",
            "extra": "mean: 403.9868042328807 usec\nrounds: 1512"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 2283.4979290960177,
            "unit": "iter/sec",
            "range": "stddev: 0.000014751821277151378",
            "extra": "mean: 437.92463626007145 usec\nrounds: 1754"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 4339.748301005441,
            "unit": "iter/sec",
            "range": "stddev: 0.000013700396327644676",
            "extra": "mean: 230.4281102589102 usec\nrounds: 3129"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2955.4482743749954,
            "unit": "iter/sec",
            "range": "stddev: 0.00002092488613536543",
            "extra": "mean: 338.3581464343088 usec\nrounds: 1837"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 10036.050050340875,
            "unit": "iter/sec",
            "range": "stddev: 0.000008260025282328054",
            "extra": "mean: 99.64079443446329 usec\nrounds: 5570"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 128.2019098454692,
            "unit": "iter/sec",
            "range": "stddev: 0.0002944265863927256",
            "extra": "mean: 7.80019580991711 msec\nrounds: 121"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 6029.227563666163,
            "unit": "iter/sec",
            "range": "stddev: 0.000008883927213639836",
            "extra": "mean: 165.85872558970635 usec\nrounds: 3349"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 14414.99896372028,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030232204282305625",
            "extra": "mean: 69.37218674221229 usec\nrounds: 10062"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 53257.36721094243,
            "unit": "iter/sec",
            "range": "stddev: 0.000003767082091330396",
            "extra": "mean: 18.776744934446118 usec\nrounds: 6712"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 18974.95776018583,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030863728458828017",
            "extra": "mean: 52.70103958272034 usec\nrounds: 15815"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 20452.765135321646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027174617254084886",
            "extra": "mean: 48.89314444202039 usec\nrounds: 18298"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 7879.626423037123,
            "unit": "iter/sec",
            "range": "stddev: 0.000005543871085769885",
            "extra": "mean: 126.90956985934875 usec\nrounds: 5189"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 113750.57705602777,
            "unit": "iter/sec",
            "range": "stddev: 0.000001154930136950081",
            "extra": "mean: 8.791164193456801 usec\nrounds: 14556"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 506539.3452607179,
            "unit": "iter/sec",
            "range": "stddev: 3.5086117551050287e-7",
            "extra": "mean: 1.9741803067347037 usec\nrounds: 95393"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1760.1088258865318,
            "unit": "iter/sec",
            "range": "stddev: 0.000008652283900970911",
            "extra": "mean: 568.1466880301108 usec\nrounds: 1061"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 259.7144361767489,
            "unit": "iter/sec",
            "range": "stddev: 0.0022128997780068342",
            "extra": "mean: 3.850382807829169 msec\nrounds: 281"
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
          "id": "1c6d3acfbd9740c3797303194d0a4f5eec665593",
          "message": "release: v0.11.1 (#54)\n\nCo-release with molrs 0.11.1 (Pyodide wheels). Pin molcrafts-molrs>=0.11.1,<0.12.",
          "timestamp": "2026-07-31T12:49:15+02:00",
          "tree_id": "ac5a127b606a66f92753cda78d496789cfed3a06",
          "url": "https://github.com/MolCrafts/molpy/commit/1c6d3acfbd9740c3797303194d0a4f5eec665593"
        },
        "date": 1785495029294,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7409.445880789514,
            "unit": "iter/sec",
            "range": "stddev: 0.00000442246573975387",
            "extra": "mean: 134.962859043576 usec\nrounds: 5186"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 111991.0829560566,
            "unit": "iter/sec",
            "range": "stddev: 0.000001001013882951673",
            "extra": "mean: 8.929282346455949 usec\nrounds: 41782"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 49252.32402642053,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017547686388127729",
            "extra": "mean: 20.303610433967904 usec\nrounds: 21085"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 31.906793491969836,
            "unit": "iter/sec",
            "range": "stddev: 0.0001274785249561154",
            "extra": "mean: 31.3412878749999 msec\nrounds: 32"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 122954.20627853203,
            "unit": "iter/sec",
            "range": "stddev: 9.8533768000022e-7",
            "extra": "mean: 8.13310931172756 usec\nrounds: 35321"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18327.869667465966,
            "unit": "iter/sec",
            "range": "stddev: 0.000002838272060691033",
            "extra": "mean: 54.561714926154934 usec\nrounds: 15168"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 360.7896141771837,
            "unit": "iter/sec",
            "range": "stddev: 0.000010945957559536703",
            "extra": "mean: 2.771698410112494 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3777.165465759085,
            "unit": "iter/sec",
            "range": "stddev: 0.00001190466247415933",
            "extra": "mean: 264.7487935239377 usec\nrounds: 2378"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 478064.4811136564,
            "unit": "iter/sec",
            "range": "stddev: 4.76302960589516e-7",
            "extra": "mean: 2.091768034451104 usec\nrounds: 60828"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 827.208594281474,
            "unit": "iter/sec",
            "range": "stddev: 0.000022050761447280297",
            "extra": "mean: 1.2088849256560434 msec\nrounds: 686"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52468.47202898186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024034963793068775",
            "extra": "mean: 19.059064640716674 usec\nrounds: 13722"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 443.5686296850668,
            "unit": "iter/sec",
            "range": "stddev: 0.000026311920342374498",
            "extra": "mean: 2.2544425666666257 msec\nrounds: 420"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217529.37287211674,
            "unit": "iter/sec",
            "range": "stddev: 7.456479666131724e-7",
            "extra": "mean: 4.597080324356424 usec\nrounds: 53894"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 124042.37608125189,
            "unit": "iter/sec",
            "range": "stddev: 9.905106868598349e-7",
            "extra": "mean: 8.06176108191419 usec\nrounds: 39005"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 152827.4113954268,
            "unit": "iter/sec",
            "range": "stddev: 8.877604402698676e-7",
            "extra": "mean: 6.5433287842100025 usec\nrounds: 61843"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 76454.21490586057,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012531360797154458",
            "extra": "mean: 13.079723612770307 usec\nrounds: 39991"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3659.183938620482,
            "unit": "iter/sec",
            "range": "stddev: 0.000014286967097288504",
            "extra": "mean: 273.28497740865186 usec\nrounds: 3010"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15555.768090673946,
            "unit": "iter/sec",
            "range": "stddev: 0.000003869120597103183",
            "extra": "mean: 64.28483596380714 usec\nrounds: 13052"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17826.52638833407,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034633646237703835",
            "extra": "mean: 56.09617814575553 usec\nrounds: 12501"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.25781544940216,
            "unit": "iter/sec",
            "range": "stddev: 0.00008426279836125292",
            "extra": "mean: 12.617052265822235 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1276.1665741675286,
            "unit": "iter/sec",
            "range": "stddev: 0.000007869723621916725",
            "extra": "mean: 783.5967656904993 usec\nrounds: 1195"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.3307729981075,
            "unit": "iter/sec",
            "range": "stddev: 0.0004466795071585505",
            "extra": "mean: 7.281689152174077 msec\nrounds: 138"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44396.449866212875,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017157972844085886",
            "extra": "mean: 22.52432352166591 usec\nrounds: 26703"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 546.8583990261676,
            "unit": "iter/sec",
            "range": "stddev: 0.000021730231518740726",
            "extra": "mean: 1.8286269384922607 msec\nrounds: 504"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 4420.1987328979,
            "unit": "iter/sec",
            "range": "stddev: 0.000006888529342177244",
            "extra": "mean: 226.2341719066545 usec\nrounds: 1972"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 108786.01146674523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010083972108962024",
            "extra": "mean: 9.192358342006958 usec\nrounds: 59446"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93534.17536610509,
            "unit": "iter/sec",
            "range": "stddev: 0.000001878455228214657",
            "extra": "mean: 10.691279375542344 usec\nrounds: 55982"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 80406.25530988365,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011940177942153836",
            "extra": "mean: 12.436843329490046 usec\nrounds: 50386"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 99398.66264957767,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010220022463192696",
            "extra": "mean: 10.060497529281887 usec\nrounds: 62736"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40041.59972375972,
            "unit": "iter/sec",
            "range": "stddev: 0.000001895438188452264",
            "extra": "mean: 24.97402718419924 usec\nrounds: 17363"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 179276.5377553382,
            "unit": "iter/sec",
            "range": "stddev: 8.522823939149321e-7",
            "extra": "mean: 5.577974745165579 usec\nrounds: 35320"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 179076.72060699933,
            "unit": "iter/sec",
            "range": "stddev: 8.461433964953436e-7",
            "extra": "mean: 5.5841987535308615 usec\nrounds: 55516"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 180183.4620887527,
            "unit": "iter/sec",
            "range": "stddev: 8.520496759358511e-7",
            "extra": "mean: 5.549898910852492 usec\nrounds: 55179"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 163259.02530644884,
            "unit": "iter/sec",
            "range": "stddev: 8.521644998629491e-7",
            "extra": "mean: 6.12523563780274 usec\nrounds: 47416"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 168231.3653116538,
            "unit": "iter/sec",
            "range": "stddev: 8.838193078208476e-7",
            "extra": "mean: 5.944194759089479 usec\nrounds: 48198"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180882.3962142798,
            "unit": "iter/sec",
            "range": "stddev: 8.944132946470338e-7",
            "extra": "mean: 5.528453961961914 usec\nrounds: 51881"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 9.032996448453568,
            "unit": "iter/sec",
            "range": "stddev: 0.00014941267589475352",
            "extra": "mean: 110.70523559999828 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 852.6433407229038,
            "unit": "iter/sec",
            "range": "stddev: 0.000020707121148302392",
            "extra": "mean: 1.1728233274561926 msec\nrounds: 794"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4192.6263073861255,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060529010811077884",
            "extra": "mean: 238.5139830464513 usec\nrounds: 3893"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 70907.21228510626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014558831932021034",
            "extra": "mean: 14.102937737548675 usec\nrounds: 27095"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1499.7926884630374,
            "unit": "iter/sec",
            "range": "stddev: 0.000025590336248015926",
            "extra": "mean: 666.7588178635431 usec\nrounds: 1142"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1335.8481641274523,
            "unit": "iter/sec",
            "range": "stddev: 0.000020052393120396794",
            "extra": "mean: 748.5880707506747 usec\nrounds: 1159"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2627.407506589306,
            "unit": "iter/sec",
            "range": "stddev: 0.000018270364605832315",
            "extra": "mean: 380.6033123876247 usec\nrounds: 2228"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1516.5999601355832,
            "unit": "iter/sec",
            "range": "stddev: 0.000023903970189385706",
            "extra": "mean: 659.36965995344 usec\nrounds: 1291"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5920.483786857365,
            "unit": "iter/sec",
            "range": "stddev: 0.000012182160066060593",
            "extra": "mean: 168.90511586567612 usec\nrounds: 3599"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 41.7546907326435,
            "unit": "iter/sec",
            "range": "stddev: 0.00010770897974740913",
            "extra": "mean: 23.949405023809874 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5804.901147618078,
            "unit": "iter/sec",
            "range": "stddev: 0.000006671772350196706",
            "extra": "mean: 172.26822207133185 usec\nrounds: 4422"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10606.921931863972,
            "unit": "iter/sec",
            "range": "stddev: 0.000014020167412484103",
            "extra": "mean: 94.27805789688398 usec\nrounds: 9966"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 30030.324954448715,
            "unit": "iter/sec",
            "range": "stddev: 0.000004257390856587198",
            "extra": "mean: 33.299672964473174 usec\nrounds: 8892"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13746.386040390511,
            "unit": "iter/sec",
            "range": "stddev: 0.00000312670696657825",
            "extra": "mean: 72.74639291096118 usec\nrounds: 9959"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14462.946216908324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031582480737765817",
            "extra": "mean: 69.14220553699643 usec\nrounds: 13798"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5273.265438922071,
            "unit": "iter/sec",
            "range": "stddev: 0.000009185148215266369",
            "extra": "mean: 189.63581704402387 usec\nrounds: 3755"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 81156.35327932492,
            "unit": "iter/sec",
            "range": "stddev: 0.00000192101459909695",
            "extra": "mean: 12.32189421520935 usec\nrounds: 19672"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 337114.69560500316,
            "unit": "iter/sec",
            "range": "stddev: 6.589102564746853e-7",
            "extra": "mean: 2.966349474042801 usec\nrounds: 88410"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1297.7097500861537,
            "unit": "iter/sec",
            "range": "stddev: 0.000020376710923010866",
            "extra": "mean: 770.5883383658102 usec\nrounds: 1126"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 192.718965049845,
            "unit": "iter/sec",
            "range": "stddev: 0.002710605292124065",
            "extra": "mean: 5.1889029174755015 msec\nrounds: 206"
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
          "id": "c602bd30ed55c7f607c3212c88f06342517e546b",
          "message": "fix(ci): use single quotes in GHA if expressions (release.yml) (#55)\n\nDouble-quoted \"push\" inside github.event_name == \"push\" is invalid\nexpression syntax and prevents the Release workflow from parsing.",
          "timestamp": "2026-07-31T12:50:41+02:00",
          "tree_id": "e874afd3290f18bfde3a04e0d7f7474f3cbedf7b",
          "url": "https://github.com/MolCrafts/molpy/commit/c602bd30ed55c7f607c3212c88f06342517e546b"
        },
        "date": 1785495118144,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 12045.782149608598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034608927488870554",
            "extra": "mean: 83.0166100947204 usec\nrounds: 4537"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 126358.34398576093,
            "unit": "iter/sec",
            "range": "stddev: 8.130718142179601e-7",
            "extra": "mean: 7.914000519923623 usec\nrounds: 34621"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 53426.31376187912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015763424775867292",
            "extra": "mean: 18.71736845736721 usec\nrounds: 15953"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 38.74995098379769,
            "unit": "iter/sec",
            "range": "stddev: 0.00013814466629495317",
            "extra": "mean: 25.806484256409117 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 129323.15283085682,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010725891828275921",
            "extra": "mean: 7.732567433674549 usec\nrounds: 26641"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 25527.05108090069,
            "unit": "iter/sec",
            "range": "stddev: 0.000002401423300846653",
            "extra": "mean: 39.1741293121084 usec\nrounds: 14407"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 373.94297852434937,
            "unit": "iter/sec",
            "range": "stddev: 0.00007386119867051991",
            "extra": "mean: 2.67420451092889 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 4971.841041102073,
            "unit": "iter/sec",
            "range": "stddev: 0.000018100713160527624",
            "extra": "mean: 201.13273769877748 usec\nrounds: 2703"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 584982.5373382348,
            "unit": "iter/sec",
            "range": "stddev: 4.3400767920486244e-7",
            "extra": "mean: 1.709452737769168 usec\nrounds: 42042"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 1286.105259370206,
            "unit": "iter/sec",
            "range": "stddev: 0.00004679194024901302",
            "extra": "mean: 777.5413347502293 usec\nrounds: 941"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 69264.10713199958,
            "unit": "iter/sec",
            "range": "stddev: 0.000001704988222151406",
            "extra": "mean: 14.437492106759668 usec\nrounds: 12479"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 688.8302942957583,
            "unit": "iter/sec",
            "range": "stddev: 0.00007370280835365304",
            "extra": "mean: 1.451736383084564 msec\nrounds: 603"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 237319.03902832823,
            "unit": "iter/sec",
            "range": "stddev: 6.351244277719229e-7",
            "extra": "mean: 4.213736934442214 usec\nrounds: 39895"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 121710.86709335653,
            "unit": "iter/sec",
            "range": "stddev: 8.575700107186279e-7",
            "extra": "mean: 8.216193211678991 usec\nrounds: 23570"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 155075.83925693762,
            "unit": "iter/sec",
            "range": "stddev: 7.609081768990655e-7",
            "extra": "mean: 6.448457766158844 usec\nrounds: 38275"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 73441.68695306756,
            "unit": "iter/sec",
            "range": "stddev: 0.000001281453249200945",
            "extra": "mean: 13.616244962333774 usec\nrounds: 24416"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 4241.896369297768,
            "unit": "iter/sec",
            "range": "stddev: 0.0000063395840866579535",
            "extra": "mean: 235.74361864138294 usec\nrounds: 3165"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 14737.732138962125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033592604611768344",
            "extra": "mean: 67.85304486273714 usec\nrounds: 8448"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 14025.150185868326,
            "unit": "iter/sec",
            "range": "stddev: 0.000003002037004812147",
            "extra": "mean: 71.30048425489198 usec\nrounds: 10670"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 95.86698167682451,
            "unit": "iter/sec",
            "range": "stddev: 0.00011130208007101883",
            "extra": "mean: 10.431120105262961 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1255.1056935636987,
            "unit": "iter/sec",
            "range": "stddev: 0.000013427208289085004",
            "extra": "mean: 796.7456486956399 usec\nrounds: 1150"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 165.26501252570802,
            "unit": "iter/sec",
            "range": "stddev: 0.000246214729606411",
            "extra": "mean: 6.050887509202491 msec\nrounds: 163"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 52696.50348361577,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013545411881009363",
            "extra": "mean: 18.976591118819048 usec\nrounds: 21889"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 654.1027443659502,
            "unit": "iter/sec",
            "range": "stddev: 0.000029278718152279736",
            "extra": "mean: 1.528811809174326 msec\nrounds: 545"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 4266.936954888298,
            "unit": "iter/sec",
            "range": "stddev: 0.000007079770104067721",
            "extra": "mean: 234.36015356504805 usec\nrounds: 1641"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 121309.73248129993,
            "unit": "iter/sec",
            "range": "stddev: 9.085827051307009e-7",
            "extra": "mean: 8.243361678784936 usec\nrounds: 36502"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 113015.63620240356,
            "unit": "iter/sec",
            "range": "stddev: 8.91509592929785e-7",
            "extra": "mean: 8.848333147539567 usec\nrounds: 46649"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 97345.64257822897,
            "unit": "iter/sec",
            "range": "stddev: 9.793189028728734e-7",
            "extra": "mean: 10.272673470683388 usec\nrounds: 48354"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 120730.68477326924,
            "unit": "iter/sec",
            "range": "stddev: 8.297576446676505e-7",
            "extra": "mean: 8.282898435289981 usec\nrounds: 60267"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 45649.50080310295,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026378920581608863",
            "extra": "mean: 21.906044587721464 usec\nrounds: 14892"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 191359.42320883993,
            "unit": "iter/sec",
            "range": "stddev: 7.55470433804443e-7",
            "extra": "mean: 5.225768259703893 usec\nrounds: 22974"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 190232.88025260088,
            "unit": "iter/sec",
            "range": "stddev: 8.283395546328418e-7",
            "extra": "mean: 5.256714815399678 usec\nrounds: 29935"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 191268.16534740146,
            "unit": "iter/sec",
            "range": "stddev: 7.485717754177101e-7",
            "extra": "mean: 5.2282615781026305 usec\nrounds: 31050"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 179435.8715456027,
            "unit": "iter/sec",
            "range": "stddev: 7.56192455631424e-7",
            "extra": "mean: 5.573021667219172 usec\nrounds: 29907"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 182505.53130223605,
            "unit": "iter/sec",
            "range": "stddev: 7.469152993219884e-7",
            "extra": "mean: 5.479285985825615 usec\nrounds: 31040"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 189749.7053824442,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011257932941633574",
            "extra": "mean: 5.270100409296977 usec\nrounds: 31272"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 10.959428338762631,
            "unit": "iter/sec",
            "range": "stddev: 0.0002895742720691352",
            "extra": "mean: 91.24563518181685 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 907.7694448468721,
            "unit": "iter/sec",
            "range": "stddev: 0.000021082496005472818",
            "extra": "mean: 1.1016012994011777 msec\nrounds: 835"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 6759.987692198895,
            "unit": "iter/sec",
            "range": "stddev: 0.000003871875542471985",
            "extra": "mean: 147.92926341478574 usec\nrounds: 5330"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 71805.10833562698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012728149776115603",
            "extra": "mean: 13.926585770553565 usec\nrounds: 24892"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 2092.6053208053017,
            "unit": "iter/sec",
            "range": "stddev: 0.000013897169343100467",
            "extra": "mean: 477.87319952678314 usec\nrounds: 1268"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1849.0934022910667,
            "unit": "iter/sec",
            "range": "stddev: 0.00006262329696795313",
            "extra": "mean: 540.8055638298089 usec\nrounds: 1410"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3653.6828594153053,
            "unit": "iter/sec",
            "range": "stddev: 0.000012425876243896822",
            "extra": "mean: 273.696442323412 usec\nrounds: 2410"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 2473.8430048819314,
            "unit": "iter/sec",
            "range": "stddev: 0.000017535585472246473",
            "extra": "mean: 404.22937026584947 usec\nrounds: 1769"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 8345.306709319584,
            "unit": "iter/sec",
            "range": "stddev: 0.00001361085003403965",
            "extra": "mean: 119.82783075944404 usec\nrounds: 4727"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 42.628149367179596,
            "unit": "iter/sec",
            "range": "stddev: 0.0003303481639610143",
            "extra": "mean: 23.45867730232557 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 4961.277122174666,
            "unit": "iter/sec",
            "range": "stddev: 0.000008432488169944044",
            "extra": "mean: 201.5610044297772 usec\nrounds: 3386"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 12327.132850228583,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038521724437227065",
            "extra": "mean: 81.12186443917953 usec\nrounds: 8889"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 45383.131951356045,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027629814228241633",
            "extra": "mean: 22.03461852460626 usec\nrounds: 5463"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 16787.741881867685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026547811223734673",
            "extra": "mean: 59.56727277776963 usec\nrounds: 14400"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 16871.675884928994,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030737845104370402",
            "extra": "mean: 59.27093472043714 usec\nrounds: 9804"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 6565.000114016981,
            "unit": "iter/sec",
            "range": "stddev: 0.000005925919435104144",
            "extra": "mean: 152.3229219546992 usec\nrounds: 4113"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 95082.48840218893,
            "unit": "iter/sec",
            "range": "stddev: 0.000001952466423915086",
            "extra": "mean: 10.5171837296696 usec\nrounds: 13079"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 405789.42083431897,
            "unit": "iter/sec",
            "range": "stddev: 5.430308102402079e-7",
            "extra": "mean: 2.4643323572703317 usec\nrounds: 79574"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1495.795468057634,
            "unit": "iter/sec",
            "range": "stddev: 0.000014861872025278556",
            "extra": "mean: 668.5406002055552 usec\nrounds: 973"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 208.00106620076284,
            "unit": "iter/sec",
            "range": "stddev: 0.00428606729279285",
            "extra": "mean: 4.807667663755142 msec\nrounds: 229"
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
          "id": "f9dcd7732ef2da9e4ac095a8fe4881bc627cedd2",
          "message": "release: v0.11.2 (#56)\n\n- requires-python >=3.14; CI/tooling on 3.14\n- ruff 0.16.1, ty 0.0.65; pin select E4/E7/E9/F for stable lint\n- pin molcrafts-molrs>=0.11.2,<0.12\n- PEP 758 except formatting (ruff target py314)",
          "timestamp": "2026-07-31T13:25:45+02:00",
          "tree_id": "1acff0000337530f3b86b30c6b9b2e896a909c4c",
          "url": "https://github.com/MolCrafts/molpy/commit/f9dcd7732ef2da9e4ac095a8fe4881bc627cedd2"
        },
        "date": 1785497321679,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7610.521051381991,
            "unit": "iter/sec",
            "range": "stddev: 0.000005918562325051857",
            "extra": "mean: 131.39704801400035 usec\nrounds: 4582"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 109089.77020951518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011349711630309773",
            "extra": "mean: 9.16676236533842 usec\nrounds: 23574"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48295.86581584666,
            "unit": "iter/sec",
            "range": "stddev: 0.000002016488336819161",
            "extra": "mean: 20.705706029021716 usec\nrounds: 17648"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 32.698737668501714,
            "unit": "iter/sec",
            "range": "stddev: 0.00013103042242114578",
            "extra": "mean: 30.582220333333765 msec\nrounds: 33"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 116928.09559680101,
            "unit": "iter/sec",
            "range": "stddev: 0.000001305897165882094",
            "extra": "mean: 8.552264491232837 usec\nrounds: 26723"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 17708.340389990877,
            "unit": "iter/sec",
            "range": "stddev: 0.000003965208389762772",
            "extra": "mean: 56.47056573213495 usec\nrounds: 9881"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 360.135951747853,
            "unit": "iter/sec",
            "range": "stddev: 0.000048705365385128074",
            "extra": "mean: 2.7767291633803444 msec\nrounds: 355"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3700.296156768824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000309029071318858",
            "extra": "mean: 270.248638928734 usec\nrounds: 2091"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 445607.138670194,
            "unit": "iter/sec",
            "range": "stddev: 7.816729946568304e-7",
            "extra": "mean: 2.2441292188097717 usec\nrounds: 65540"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 804.6492131560232,
            "unit": "iter/sec",
            "range": "stddev: 0.00006851647092455759",
            "extra": "mean: 1.2427775776698584 msec\nrounds: 618"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 54568.138342375285,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034308400664579845",
            "extra": "mean: 18.325712226532797 usec\nrounds: 11516"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 435.6673468286069,
            "unit": "iter/sec",
            "range": "stddev: 0.00005072429243119202",
            "extra": "mean: 2.295329239795893 msec\nrounds: 392"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 194519.9411179193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011980751320245187",
            "extra": "mean: 5.140861107878875 usec\nrounds: 43379"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 115968.91373588491,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012701207650226545",
            "extra": "mean: 8.623000490264698 usec\nrounds: 24476"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 147986.43286187368,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011314228989211423",
            "extra": "mean: 6.757376204434711 usec\nrounds: 39957"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75138.90244996657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015500976201751239",
            "extra": "mean: 13.308685213573343 usec\nrounds: 27444"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3770.0267842254334,
            "unit": "iter/sec",
            "range": "stddev: 0.000009385115153754566",
            "extra": "mean: 265.2501049022265 usec\nrounds: 2917"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15643.626249753212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036007715328216473",
            "extra": "mean: 63.92379772022333 usec\nrounds: 10001"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17694.14425159123,
            "unit": "iter/sec",
            "range": "stddev: 0.000003125614432442619",
            "extra": "mean: 56.51587247063785 usec\nrounds: 11762"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.98297111345548,
            "unit": "iter/sec",
            "range": "stddev: 0.00011276789175699328",
            "extra": "mean: 12.660956987342816 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1260.6939262030694,
            "unit": "iter/sec",
            "range": "stddev: 0.000009770590784006061",
            "extra": "mean: 793.2139429050619 usec\nrounds: 1191"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 138.1635639680655,
            "unit": "iter/sec",
            "range": "stddev: 0.000029060471762994968",
            "extra": "mean: 7.237798239130076 msec\nrounds: 138"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44312.455949428666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019633471412227905",
            "extra": "mean: 22.567018202314134 usec\nrounds: 24063"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 674.5929840684186,
            "unit": "iter/sec",
            "range": "stddev: 0.00001955138432865043",
            "extra": "mean: 1.4823753338925594 msec\nrounds: 596"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 4349.090007622404,
            "unit": "iter/sec",
            "range": "stddev: 0.000009684562206319504",
            "extra": "mean: 229.93315802785332 usec\nrounds: 1582"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 107386.47099610092,
            "unit": "iter/sec",
            "range": "stddev: 0.000001185133680671739",
            "extra": "mean: 9.312160002318251 usec\nrounds: 52643"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93824.48406497976,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013146994385655543",
            "extra": "mean: 10.658198762994878 usec\nrounds: 46402"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 78458.92388489266,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020660979263383345",
            "extra": "mean: 12.745522758725357 usec\nrounds: 46993"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 103558.73940070944,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019492215258467987",
            "extra": "mean: 9.656355473106014 usec\nrounds: 57664"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40205.66895724516,
            "unit": "iter/sec",
            "range": "stddev: 0.000002236522757735739",
            "extra": "mean: 24.872114453894632 usec\nrounds: 15456"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 174599.78101531006,
            "unit": "iter/sec",
            "range": "stddev: 9.87460134368424e-7",
            "extra": "mean: 5.727384044727486 usec\nrounds: 23453"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 175217.5061058818,
            "unit": "iter/sec",
            "range": "stddev: 9.739397013009163e-7",
            "extra": "mean: 5.707192290453628 usec\nrounds: 24489"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 174702.14109121825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010308000213481351",
            "extra": "mean: 5.724028301850429 usec\nrounds: 24168"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 158401.94942773227,
            "unit": "iter/sec",
            "range": "stddev: 9.85970049599802e-7",
            "extra": "mean: 6.313053618422985 usec\nrounds: 33813"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 164355.395940899,
            "unit": "iter/sec",
            "range": "stddev: 0.000001149148242330522",
            "extra": "mean: 6.084375838561409 usec\nrounds: 18633"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 175528.08037889565,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012322077845284671",
            "extra": "mean: 5.697094150641855 usec\nrounds: 25353"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 8.842094708116491,
            "unit": "iter/sec",
            "range": "stddev: 0.00696032439441249",
            "extra": "mean: 113.09537310000337 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 847.7182822152506,
            "unit": "iter/sec",
            "range": "stddev: 0.00001501607943653018",
            "extra": "mean: 1.179637175438529 msec\nrounds: 798"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4228.658637246688,
            "unit": "iter/sec",
            "range": "stddev: 0.000007465691293691915",
            "extra": "mean: 236.48160936705634 usec\nrounds: 3950"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 70300.1505585745,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017980504068537085",
            "extra": "mean: 14.224720602366192 usec\nrounds: 19789"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1457.2402509195313,
            "unit": "iter/sec",
            "range": "stddev: 0.000023343993913281853",
            "extra": "mean: 686.2286430593659 usec\nrounds: 1059"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1317.7300463613701,
            "unit": "iter/sec",
            "range": "stddev: 0.000022964734340302683",
            "extra": "mean: 758.8807758928214 usec\nrounds: 1120"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2560.8597543187607,
            "unit": "iter/sec",
            "range": "stddev: 0.000021060639497111053",
            "extra": "mean: 390.49385594566456 usec\nrounds: 1909"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1451.227983657869,
            "unit": "iter/sec",
            "range": "stddev: 0.00005727372025340814",
            "extra": "mean: 689.0716078113835 usec\nrounds: 1229"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5605.871015532939,
            "unit": "iter/sec",
            "range": "stddev: 0.000014922043137775618",
            "extra": "mean: 178.38441113417804 usec\nrounds: 3826"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 39.189455938208475,
            "unit": "iter/sec",
            "range": "stddev: 0.0001706400475272296",
            "extra": "mean: 25.517067692308324 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 6177.525308979872,
            "unit": "iter/sec",
            "range": "stddev: 0.000009332084561821283",
            "extra": "mean: 161.8771190700529 usec\nrounds: 3183"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11098.880447374455,
            "unit": "iter/sec",
            "range": "stddev: 0.000006317403471978219",
            "extra": "mean: 90.0991775469173 usec\nrounds: 8471"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29021.46680978306,
            "unit": "iter/sec",
            "range": "stddev: 0.000005042041768297157",
            "extra": "mean: 34.45725216283357 usec\nrounds: 6242"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13842.996167069965,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031599496123925964",
            "extra": "mean: 72.23869658931372 usec\nrounds: 12109"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14429.26486187758,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035678925319893835",
            "extra": "mean: 69.30359998048279 usec\nrounds: 10297"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5279.485319102711,
            "unit": "iter/sec",
            "range": "stddev: 0.000008799694850475767",
            "extra": "mean: 189.41240283057698 usec\nrounds: 3957"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 77727.08506500156,
            "unit": "iter/sec",
            "range": "stddev: 0.000002120878130078915",
            "extra": "mean: 12.86552813814799 usec\nrounds: 12794"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 336823.7251681148,
            "unit": "iter/sec",
            "range": "stddev: 7.474636348422129e-7",
            "extra": "mean: 2.968912001376631 usec\nrounds: 70092"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1421.0981409933863,
            "unit": "iter/sec",
            "range": "stddev: 0.00001701079040068744",
            "extra": "mean: 703.6811682133176 usec\nrounds: 862"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 208.25020270112583,
            "unit": "iter/sec",
            "range": "stddev: 0.0032433971165198253",
            "extra": "mean: 4.801916094339503 msec\nrounds: 212"
          }
        ]
      }
    ]
  }
}