window.BENCHMARK_DATA = {
  "lastUpdate": 1782916300541,
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
      }
    ]
  }
}