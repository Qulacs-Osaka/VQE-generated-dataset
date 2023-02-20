OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.5721116070096184) q[0];
rz(-2.4604740052893685) q[0];
ry(0.2579613130657261) q[1];
rz(-2.7814370122189715) q[1];
ry(-0.0004460619930846121) q[2];
rz(-2.1178944440076504) q[2];
ry(-1.5726143799719818) q[3];
rz(2.525796570012093) q[3];
ry(-1.535807909476118) q[4];
rz(-2.844588745509568) q[4];
ry(3.141338406599303) q[5];
rz(0.48804135386065184) q[5];
ry(-1.9103012081872892) q[6];
rz(1.5125225947975816) q[6];
ry(-1.1305992711908894) q[7];
rz(-0.17817412567969002) q[7];
ry(0.0003571401587710473) q[8];
rz(-0.5737147679594905) q[8];
ry(-0.7506842135689215) q[9];
rz(-2.0231494705707824) q[9];
ry(2.9860712421591886) q[10];
rz(1.1502475472592766) q[10];
ry(-2.458163243785731) q[11];
rz(0.2967239613024265) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.1050560075226175) q[0];
rz(1.6564547827595353) q[0];
ry(-2.823368876041462) q[1];
rz(-1.5482100599260202) q[1];
ry(-1.1914335055508642) q[2];
rz(0.09528191407313889) q[2];
ry(-2.743328444555227) q[3];
rz(-0.31528376758863974) q[3];
ry(-2.8189063930199194) q[4];
rz(-2.1321835480693174) q[4];
ry(-2.5667654296517237) q[5];
rz(2.6634198632929853) q[5];
ry(2.1238751964093767) q[6];
rz(-2.431910384855861) q[6];
ry(2.685962070535804) q[7];
rz(1.5348563105779676) q[7];
ry(9.850690382460276e-06) q[8];
rz(-1.123415464728827) q[8];
ry(-0.6903383150416156) q[9];
rz(2.8128934027798054) q[9];
ry(2.9447060137143) q[10];
rz(-1.3842124825802937) q[10];
ry(-0.9707789143393611) q[11];
rz(-2.0906789375623513) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5145507248983971) q[0];
rz(1.3631954322176352) q[0];
ry(-2.4954753199874125) q[1];
rz(-2.8837644300127354) q[1];
ry(1.1776342331806324) q[2];
rz(1.2326496606382662) q[2];
ry(-1.3307552967804532) q[3];
rz(-0.6233814590731194) q[3];
ry(2.5373840779734413) q[4];
rz(-2.1176923624655224) q[4];
ry(-1.914226614567963) q[5];
rz(2.455620843445695) q[5];
ry(0.6798619600072708) q[6];
rz(0.5389565881277258) q[6];
ry(-1.813904849992661) q[7];
rz(-2.0086538437935157) q[7];
ry(-3.141438436278724) q[8];
rz(-2.275139047163208) q[8];
ry(2.961950191198447) q[9];
rz(-1.7479006215787323) q[9];
ry(1.9742821902797283) q[10];
rz(0.038114358632642016) q[10];
ry(1.148817230651692) q[11];
rz(-1.1582519129388338) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.203566380212544) q[0];
rz(-1.3393417659263402) q[0];
ry(-1.025098760094834) q[1];
rz(0.3473665808825584) q[1];
ry(2.2525610779758747) q[2];
rz(-2.4170570390874455) q[2];
ry(-3.1399062320707296) q[3];
rz(-0.9195931611717524) q[3];
ry(3.140904808919818) q[4];
rz(0.13798841382728444) q[4];
ry(1.3751550180040144) q[5];
rz(-2.395935280070763) q[5];
ry(1.5389028540021166) q[6];
rz(-1.2427151742661655) q[6];
ry(-1.9500859645666386) q[7];
rz(1.3843006831109794) q[7];
ry(3.1413808372854093) q[8];
rz(1.7732808802366016) q[8];
ry(1.0844019155084905) q[9];
rz(0.051049417082910505) q[9];
ry(1.093732362798404) q[10];
rz(1.6262877195019652) q[10];
ry(1.5815513123805327) q[11];
rz(-0.9599205380486752) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.7009145658520897) q[0];
rz(1.5657084920995092) q[0];
ry(2.7996347269282222) q[1];
rz(-2.792874074777658) q[1];
ry(3.141202482179623) q[2];
rz(-0.2530110308577593) q[2];
ry(-0.6211620330726614) q[3];
rz(-2.7221431488082053) q[3];
ry(-2.9556353884384325) q[4];
rz(-1.3314687482959817) q[4];
ry(-2.2007716639044723) q[5];
rz(-0.19937074312544967) q[5];
ry(-3.0156406865966323) q[6];
rz(-1.3888220218235214) q[6];
ry(0.11229487704739981) q[7];
rz(-1.6756124575627105) q[7];
ry(7.052585508354656e-05) q[8];
rz(-3.1104812900180994) q[8];
ry(-1.0296682451300334) q[9];
rz(-1.406828840747418) q[9];
ry(0.15171383737528643) q[10];
rz(-2.3919143216768526) q[10];
ry(0.18526048416087004) q[11];
rz(-2.6805161884285464) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5069757744748653) q[0];
rz(-2.5325951451966904) q[0];
ry(2.037672901142841) q[1];
rz(0.803622852113322) q[1];
ry(0.9686550742137784) q[2];
rz(0.04081153489283907) q[2];
ry(0.0011059433989829125) q[3];
rz(-0.9069403032279396) q[3];
ry(0.00014423200057134264) q[4];
rz(-1.7623533665176545) q[4];
ry(1.9991629123052226) q[5];
rz(0.7861625235893461) q[5];
ry(-2.0697584097211026) q[6];
rz(-0.25842096836477385) q[6];
ry(-2.6652360004701094) q[7];
rz(-1.0950796829372198) q[7];
ry(7.065645157525136e-05) q[8];
rz(1.368625867497573) q[8];
ry(-2.5678452684011717) q[9];
rz(-0.024436393677368606) q[9];
ry(2.859411802376866) q[10];
rz(0.11145262471455375) q[10];
ry(2.7418113729793063) q[11];
rz(-1.8959634907935516) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.489589792381949) q[0];
rz(0.8419149176279577) q[0];
ry(-2.31431394986409) q[1];
rz(1.9675762926885603) q[1];
ry(1.8515966026325161) q[2];
rz(-0.5222461527797896) q[2];
ry(0.5034719705573796) q[3];
rz(0.519442035358719) q[3];
ry(2.1243368172300827) q[4];
rz(-1.0726882614655666) q[4];
ry(1.6118161932281838) q[5];
rz(-3.104555737683028) q[5];
ry(-2.3749903237035217) q[6];
rz(0.534015841465452) q[6];
ry(0.43354578214044975) q[7];
rz(0.228977718332105) q[7];
ry(-4.569819089740093e-05) q[8];
rz(1.0690900831698409) q[8];
ry(2.9860837884467437) q[9];
rz(-1.7943135653179771) q[9];
ry(-3.0711169747972127) q[10];
rz(-1.7699628334671296) q[10];
ry(1.017554445786229) q[11];
rz(-2.0074650095245197) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.21311731810553) q[0];
rz(0.6800090341347769) q[0];
ry(3.0955931149925284) q[1];
rz(-1.6272518967735135) q[1];
ry(2.191137616925952) q[2];
rz(-0.21676319775972794) q[2];
ry(-0.0006743016801991075) q[3];
rz(-2.842312893702807) q[3];
ry(-3.1413366823162128) q[4];
rz(-2.670248010106046) q[4];
ry(-1.0845488905993004) q[5];
rz(0.3380769218427256) q[5];
ry(1.8743695378922345) q[6];
rz(-2.169461865698592) q[6];
ry(-0.6126407800975473) q[7];
rz(-1.026319546781448) q[7];
ry(-4.239575195263967e-05) q[8];
rz(0.4906283523923508) q[8];
ry(2.861421982813743) q[9];
rz(-0.34701249358336295) q[9];
ry(-1.7184528684489673) q[10];
rz(-2.3299886163007413) q[10];
ry(0.9966513426551247) q[11];
rz(0.32364684970366314) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.7000297740889625) q[0];
rz(-2.2118177764763405) q[0];
ry(3.0560355180343306) q[1];
rz(-2.5903323768694766) q[1];
ry(-0.6763530430630161) q[2];
rz(-1.5629961482935368) q[2];
ry(1.1271482987768806) q[3];
rz(-2.025816249275276) q[3];
ry(-1.6512845739875768) q[4];
rz(2.9238158047256073) q[4];
ry(0.9788469221959601) q[5];
rz(0.34890557201172795) q[5];
ry(-0.47133078630701386) q[6];
rz(-0.6818533914205719) q[6];
ry(3.0123406344077255) q[7];
rz(2.0204838930031985) q[7];
ry(3.1414619530191) q[8];
rz(-0.6063509780032772) q[8];
ry(2.916676423542004) q[9];
rz(3.113407695085103) q[9];
ry(2.004833335911187) q[10];
rz(2.618274743264391) q[10];
ry(2.115978251263738) q[11];
rz(2.911491706237146) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.2832221915151054) q[0];
rz(-0.6169291866031783) q[0];
ry(2.8754424459031585) q[1];
rz(1.5965897584334972) q[1];
ry(0.3147404320154683) q[2];
rz(-2.277823269886615) q[2];
ry(-1.3983814770535752) q[3];
rz(-2.8938050154736965) q[3];
ry(-2.8524254786567984) q[4];
rz(0.3985363226771419) q[4];
ry(0.17421129830598678) q[5];
rz(1.792019636853487) q[5];
ry(-1.2406254931711806) q[6];
rz(-2.393626230992963) q[6];
ry(-0.12180922320247588) q[7];
rz(-1.5651701847039476) q[7];
ry(-5.5729648006381825e-05) q[8];
rz(0.8799336644989646) q[8];
ry(-0.3607391961734008) q[9];
rz(-2.2522451373926637) q[9];
ry(0.402192290621022) q[10];
rz(-0.2768206311663748) q[10];
ry(-1.0305080269347204) q[11];
rz(1.0367313083931944) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.28360319332252) q[0];
rz(-1.7525428286286053) q[0];
ry(0.7136418792498453) q[1];
rz(0.5438904531144502) q[1];
ry(-0.5109325960804645) q[2];
rz(-2.545798645429768) q[2];
ry(0.029355533363590425) q[3];
rz(2.648565862450521) q[3];
ry(-3.141264291041602) q[4];
rz(1.8216084411499356) q[4];
ry(-3.138937075296304) q[5];
rz(-1.593464842367653) q[5];
ry(-2.8360001897921334) q[6];
rz(1.0456737497200743) q[6];
ry(-0.6064192476420338) q[7];
rz(1.321649797157375) q[7];
ry(-3.1413095899203514) q[8];
rz(-0.9443880045910403) q[8];
ry(0.9003942913520597) q[9];
rz(-0.8453316300347522) q[9];
ry(2.3462009878708736) q[10];
rz(-0.024325371494654805) q[10];
ry(-2.670614208270866) q[11];
rz(-3.0273256097169465) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.9411128788793814) q[0];
rz(-1.1792861850963279) q[0];
ry(0.00028821302132353566) q[1];
rz(-2.5557839442059587) q[1];
ry(-3.1399086677122297) q[2];
rz(0.6349280083452689) q[2];
ry(-0.1964947902509682) q[3];
rz(1.7866667855810627) q[3];
ry(-1.4161994913153555) q[4];
rz(-0.9426903644143461) q[4];
ry(-2.305086671853474) q[5];
rz(0.5945439631112523) q[5];
ry(-0.15303533167161107) q[6];
rz(2.118137355531565) q[6];
ry(2.0163822425054) q[7];
rz(1.9682210479070703) q[7];
ry(1.2796272343079418) q[8];
rz(1.9888505123307976) q[8];
ry(-0.04182058499285048) q[9];
rz(0.621279810189676) q[9];
ry(-0.34830462189764066) q[10];
rz(-1.4568705181636499) q[10];
ry(-1.4541606482543559) q[11];
rz(-2.9254573220084166) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.303665645630191) q[0];
rz(0.8156073010928645) q[0];
ry(-1.5924665683304868) q[1];
rz(0.7544410081130728) q[1];
ry(2.6173840519793776) q[2];
rz(2.8342806008778303) q[2];
ry(-1.109551136787463) q[3];
rz(0.1956289708052794) q[3];
ry(-0.1607517157549716) q[4];
rz(2.3032470243893193) q[4];
ry(0.010276461285368512) q[5];
rz(2.0658734001311583) q[5];
ry(-2.273615864902757) q[6];
rz(2.4355034253305052) q[6];
ry(0.00010829071546414326) q[7];
rz(-0.3416579514746756) q[7];
ry(-1.2256358246531818e-05) q[8];
rz(1.1543902770747883) q[8];
ry(2.1426904514742704) q[9];
rz(-1.6561809538428298) q[9];
ry(3.0042125433334212) q[10];
rz(0.6779943105201557) q[10];
ry(2.933888104722673) q[11];
rz(-0.17464034158822983) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.1961025448007483) q[0];
rz(-2.3403587282659277) q[0];
ry(-2.8743729297820786) q[1];
rz(1.323621435660538) q[1];
ry(-0.0009763800350317008) q[2];
rz(-2.5297546290302306) q[2];
ry(-2.367413445826432) q[3];
rz(-0.46544126480014825) q[3];
ry(0.0010881140014857849) q[4];
rz(2.6806621726340034) q[4];
ry(3.138305883949394) q[5];
rz(2.306096582625124) q[5];
ry(-0.396550408048642) q[6];
rz(-0.25121412480708016) q[6];
ry(2.913083139828536) q[7];
rz(1.6524900088743575) q[7];
ry(-2.260350962291972) q[8];
rz(0.2874361951726509) q[8];
ry(3.1408033742548622) q[9];
rz(1.4830932531681107) q[9];
ry(-0.0003743111300884236) q[10];
rz(0.3098034612492308) q[10];
ry(-0.6288900237867363) q[11];
rz(-1.8315523669491511) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.7718359271551531) q[0];
rz(0.19051643326605117) q[0];
ry(-3.035796955723856) q[1];
rz(-1.21913968958206) q[1];
ry(-0.009665245205504316) q[2];
rz(3.0580847624990226) q[2];
ry(0.998479994155411) q[3];
rz(-0.21750967841373453) q[3];
ry(1.6712501734573115) q[4];
rz(-1.7500013103479108) q[4];
ry(-3.0764195113125448) q[5];
rz(2.5695680954675475) q[5];
ry(0.8271047344564817) q[6];
rz(-2.2593035029331476) q[6];
ry(-7.176365576455623e-05) q[7];
rz(1.1627987298140627) q[7];
ry(3.1411634028772464) q[8];
rz(1.5946438345554452) q[8];
ry(0.9722523257129918) q[9];
rz(2.4063983261628747) q[9];
ry(1.9999246972516822) q[10];
rz(2.2019156069679306) q[10];
ry(-1.0306047857880491) q[11];
rz(2.123508249244309) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.9522092116697536) q[0];
rz(2.4773209330096266) q[0];
ry(1.019885916786426) q[1];
rz(-2.746194489490942) q[1];
ry(3.136855717706082) q[2];
rz(-1.1269995673600937) q[2];
ry(-0.33452356358685975) q[3];
rz(-2.4443571483760898) q[3];
ry(3.0868453700888208) q[4];
rz(1.2877337985544903) q[4];
ry(-0.004051596523315502) q[5];
rz(0.567241778741118) q[5];
ry(-1.4915372277938683) q[6];
rz(-1.772913716792158) q[6];
ry(3.0875846294712197) q[7];
rz(1.5738377695123988) q[7];
ry(2.0025660579136746) q[8];
rz(0.7401651649550525) q[8];
ry(1.2674892425425242) q[9];
rz(0.5960977873395029) q[9];
ry(0.12965406383391584) q[10];
rz(-2.6965839171380477) q[10];
ry(2.760614268916877) q[11];
rz(2.4665216745652523) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.386292274977178) q[0];
rz(-2.3982473164637206) q[0];
ry(-0.7028952157028114) q[1];
rz(3.0505814585086886) q[1];
ry(0.0023903872238659335) q[2];
rz(-0.9928004840691359) q[2];
ry(2.591105728409956) q[3];
rz(0.6645380996795144) q[3];
ry(2.237486624332906) q[4];
rz(1.239224536999609) q[4];
ry(-0.1627464358231247) q[5];
rz(1.6834641078939216) q[5];
ry(-1.4184599083793055) q[6];
rz(-2.0730417067130666) q[6];
ry(3.042423814415055) q[7];
rz(3.098976445892678) q[7];
ry(-0.018242234408186057) q[8];
rz(2.0457523874196673) q[8];
ry(2.8362509539574328) q[9];
rz(0.38889467037256104) q[9];
ry(0.1399683219431811) q[10];
rz(-1.011341333660009) q[10];
ry(2.118401547873738) q[11];
rz(0.27709806123164704) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.9747355770635426) q[0];
rz(0.3093404764701937) q[0];
ry(-1.5172273040038953) q[1];
rz(-1.1766654709010016) q[1];
ry(3.1387867302192376) q[2];
rz(1.5450105855007727) q[2];
ry(0.7032163146941599) q[3];
rz(0.3644030732895625) q[3];
ry(0.3148753826707999) q[4];
rz(-1.998731376435389) q[4];
ry(0.0006180177497426512) q[5];
rz(1.353842338775694) q[5];
ry(-0.18508583175484047) q[6];
rz(-2.1595091791374186) q[6];
ry(-0.004069407421080018) q[7];
rz(0.14843945199127478) q[7];
ry(-0.0017146788053548079) q[8];
rz(1.6151079283455136) q[8];
ry(3.134037926931692) q[9];
rz(-2.770921078304103) q[9];
ry(-0.0007576046269820427) q[10];
rz(1.2389039753386026) q[10];
ry(-2.765577637454813) q[11];
rz(-0.04791508607767316) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.08464736398685) q[0];
rz(2.0507747779974936) q[0];
ry(2.8323607482436923) q[1];
rz(-3.1147621914926225) q[1];
ry(-3.1400119443362318) q[2];
rz(-1.3617074805392901) q[2];
ry(2.5519396399284275) q[3];
rz(-1.7734495424759549) q[3];
ry(0.2496445941715235) q[4];
rz(-1.2841182022081152) q[4];
ry(-0.023117974735476315) q[5];
rz(-1.4872087969053365) q[5];
ry(-2.3129939889728663) q[6];
rz(-2.1675801590671493) q[6];
ry(3.011935819910268) q[7];
rz(-1.5665814124235715) q[7];
ry(-1.5556291939057383) q[8];
rz(-0.008788293202611543) q[8];
ry(2.8267549753334573) q[9];
rz(0.7620237996623396) q[9];
ry(-1.6338175432500597) q[10];
rz(0.857743385242527) q[10];
ry(-2.3633136110289956) q[11];
rz(2.87157961042907) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.2464402484902317) q[0];
rz(1.8239170619091223) q[0];
ry(-1.9011112016831468) q[1];
rz(-2.4428296955884448) q[1];
ry(3.0778775367836846) q[2];
rz(-1.500998755182441) q[2];
ry(-1.2034917539523762) q[3];
rz(0.37505347901902797) q[3];
ry(1.2914787829694845) q[4];
rz(-1.3688351129176106) q[4];
ry(-3.121651702229077) q[5];
rz(0.9618208954894518) q[5];
ry(-0.759836596236356) q[6];
rz(-1.7749466039785011) q[6];
ry(-0.008725519789550162) q[7];
rz(1.600694952552452) q[7];
ry(1.4223741128147902) q[8];
rz(-2.7562124462348128) q[8];
ry(1.5705875889725078) q[9];
rz(0.03268814453193531) q[9];
ry(0.018863533778711992) q[10];
rz(-1.3773194106854083) q[10];
ry(-1.6874889285599686) q[11];
rz(1.9956083502607624) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3096433063553414) q[0];
rz(3.0559634441876504) q[0];
ry(0.6009106500582986) q[1];
rz(2.116105672455525) q[1];
ry(0.9573153077388058) q[2];
rz(-0.38226213303988443) q[2];
ry(1.2951903253875516) q[3];
rz(-1.9352315636195296) q[3];
ry(-1.494613538304697) q[4];
rz(-1.7903920190672036) q[4];
ry(3.1315450494723223) q[5];
rz(-2.2058320353217966) q[5];
ry(0.10416015914848595) q[6];
rz(1.4817307268136393) q[6];
ry(-2.0133372233765705) q[7];
rz(1.482947755071478) q[7];
ry(2.5165429303691287) q[8];
rz(-0.20548390954534046) q[8];
ry(-0.17721378471005886) q[9];
rz(0.326713902821571) q[9];
ry(-1.8099340470546683) q[10];
rz(2.9531944642671974) q[10];
ry(1.2718651690015417) q[11];
rz(2.1559791455121715) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.0423353897158254) q[0];
rz(-2.154293423524007) q[0];
ry(0.08293698257776683) q[1];
rz(-0.6696309896025685) q[1];
ry(1.0070164570647542) q[2];
rz(-3.1044487060489936) q[2];
ry(0.002730648005727865) q[3];
rz(-1.7151703029972514) q[3];
ry(0.0505214851254383) q[4];
rz(1.8690137782029574) q[4];
ry(-0.10041622859231848) q[5];
rz(-0.5674889444253391) q[5];
ry(2.0737612009381072) q[6];
rz(0.1376659373791318) q[6];
ry(3.124523557889587) q[7];
rz(1.7246027148607208) q[7];
ry(2.4519873677401383) q[8];
rz(-0.13018653293817373) q[8];
ry(-3.13831642010815) q[9];
rz(-1.4051292648405642) q[9];
ry(-3.1246369588574474) q[10];
rz(-2.648089696914316) q[10];
ry(2.930781456615739) q[11];
rz(0.7687586432837366) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.4058072127534356) q[0];
rz(-2.4397139847894818) q[0];
ry(0.23157705051186891) q[1];
rz(0.7074544910929381) q[1];
ry(1.266697082945367) q[2];
rz(1.5360254558595754) q[2];
ry(-0.024647950798434515) q[3];
rz(-2.4049777241379737) q[3];
ry(-2.366088191271484) q[4];
rz(2.8924987352492155) q[4];
ry(2.819933917357498) q[5];
rz(-2.816318825638336) q[5];
ry(-0.7549822687806412) q[6];
rz(1.4797124831384272) q[6];
ry(-0.1543991024265254) q[7];
rz(2.787410393310533) q[7];
ry(-2.1107497650723825) q[8];
rz(-1.8831056812306404) q[8];
ry(-0.0032922334144946026) q[9];
rz(-0.43058853880412007) q[9];
ry(-2.6351902076962297) q[10];
rz(-1.1179244243483393) q[10];
ry(1.1935453967436827) q[11];
rz(0.004231009214400672) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.4309317820663444) q[0];
rz(2.2703343278458554) q[0];
ry(2.770270126954007) q[1];
rz(2.567461492936575) q[1];
ry(0.6672403042682076) q[2];
rz(1.9929766171940173) q[2];
ry(0.444452988581418) q[3];
rz(2.73437989812782) q[3];
ry(-3.0220858134825868) q[4];
rz(1.619971183457035) q[4];
ry(-0.2150230674354409) q[5];
rz(-2.386435222742388) q[5];
ry(-0.9303673837727476) q[6];
rz(0.5398900098538029) q[6];
ry(-0.01726934512519307) q[7];
rz(0.04908651797270495) q[7];
ry(-0.5405568261406092) q[8];
rz(-1.4378782911559915) q[8];
ry(0.0011227624887526488) q[9];
rz(-1.9523911924429536) q[9];
ry(-0.0708090143653397) q[10];
rz(1.727963263102425) q[10];
ry(2.3069472104647435) q[11];
rz(2.58653201153329) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.015201338053222) q[0];
rz(2.2640068971480045) q[0];
ry(-3.116507435486034) q[1];
rz(2.983297714410118) q[1];
ry(0.0036839138086341224) q[2];
rz(2.531495603515616) q[2];
ry(3.123708869592039) q[3];
rz(-2.2333584347272053) q[3];
ry(3.0206816446617593) q[4];
rz(-1.0605980815835707) q[4];
ry(-3.1149909821575585) q[5];
rz(2.4668733620920675) q[5];
ry(2.9080957695393836) q[6];
rz(-3.0187880871971693) q[6];
ry(0.00855645792085191) q[7];
rz(0.058829480191473056) q[7];
ry(2.8647754796556644) q[8];
rz(1.409017915294589) q[8];
ry(3.1405586695718046) q[9];
rz(-1.5403654047790944) q[9];
ry(-1.7063039675253648) q[10];
rz(0.29723476445228414) q[10];
ry(0.09690575273503027) q[11];
rz(-0.6569351707332375) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.2641389407348171) q[0];
rz(-0.9324179775531467) q[0];
ry(2.8822798066173196) q[1];
rz(-0.004438172954185582) q[1];
ry(-0.4680096363936137) q[2];
rz(-1.2354086025770972) q[2];
ry(-2.782623123282775) q[3];
rz(-2.1129625657335094) q[3];
ry(-2.93667615070175) q[4];
rz(-2.656251130318183) q[4];
ry(2.537850961955763) q[5];
rz(2.586877384543694) q[5];
ry(1.1182741039459154) q[6];
rz(-0.7324091148991556) q[6];
ry(-1.7924065020735307) q[7];
rz(0.12231284224393683) q[7];
ry(-0.41875092884684806) q[8];
rz(2.3093811082608773) q[8];
ry(-3.1403456841633415) q[9];
rz(-1.9319990100088913) q[9];
ry(0.36403367291825667) q[10];
rz(3.0680417619533027) q[10];
ry(1.1455606571528665) q[11];
rz(1.9658895340691032) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.8832605141015737) q[0];
rz(2.4377248194747647) q[0];
ry(-0.6729715213084303) q[1];
rz(-0.5318287765120562) q[1];
ry(0.007572968884678176) q[2];
rz(0.20863272002430233) q[2];
ry(-2.985069292383817) q[3];
rz(-1.8031563625696327) q[3];
ry(3.134853835786492) q[4];
rz(1.1343723068444167) q[4];
ry(-3.118824305006402) q[5];
rz(1.0001770832577128) q[5];
ry(-0.003912777745244789) q[6];
rz(-0.4167342068842492) q[6];
ry(0.02234455614370873) q[7];
rz(-0.045762785589769484) q[7];
ry(0.0036980827605432864) q[8];
rz(1.9912751423216903) q[8];
ry(0.0021185644583079366) q[9];
rz(2.2769060752448462) q[9];
ry(1.9319024744342652) q[10];
rz(-1.308016755388472) q[10];
ry(0.019445675620425644) q[11];
rz(1.662044698398307) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.3270544183243884) q[0];
rz(-1.4863386842135933) q[0];
ry(3.127820763620272) q[1];
rz(-0.3674867632196444) q[1];
ry(0.002683329663761391) q[2];
rz(-2.118722822921156) q[2];
ry(2.0779198818270332) q[3];
rz(-2.932053953640488) q[3];
ry(-3.049177747378526) q[4];
rz(2.0505879941823117) q[4];
ry(0.12555746949245902) q[5];
rz(-0.5071065962582111) q[5];
ry(2.090931481783799) q[6];
rz(-1.0737994043427972) q[6];
ry(1.8749996036452101) q[7];
rz(0.035625188741314595) q[7];
ry(-0.8839657781850283) q[8];
rz(1.4202177501718447) q[8];
ry(-1.8654400977333578) q[9];
rz(0.44835875655287705) q[9];
ry(1.504347055737363) q[10];
rz(-1.1784716152918515) q[10];
ry(-0.7765895037004027) q[11];
rz(-0.7411226620416193) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.1423403686108191) q[0];
rz(-1.9230832817630767) q[0];
ry(-2.194247123410947) q[1];
rz(1.7007643939736017) q[1];
ry(0.7845173263181504) q[2];
rz(1.614812276838487) q[2];
ry(1.1377510513470594) q[3];
rz(-1.0067257704222146) q[3];
ry(3.1375970907585) q[4];
rz(2.9897114264138613) q[4];
ry(-3.130226736850319) q[5];
rz(-1.7862066714167442) q[5];
ry(0.04670561660181535) q[6];
rz(-2.2571237811032887) q[6];
ry(-0.08161588627801672) q[7];
rz(-0.327991531140321) q[7];
ry(3.1402818960519623) q[8];
rz(-2.4569751880601536) q[8];
ry(3.141239149537528) q[9];
rz(0.4518798903660659) q[9];
ry(3.141517838940846) q[10];
rz(-1.9530124003241378) q[10];
ry(2.5303851682635536) q[11];
rz(3.0756394873798567) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.932512467530398) q[0];
rz(2.1487419868971536) q[0];
ry(0.4176994417514458) q[1];
rz(0.08088440430192279) q[1];
ry(-3.1275435492377412) q[2];
rz(1.5396092629643379) q[2];
ry(1.3847538881933863) q[3];
rz(-2.951502212616007) q[3];
ry(0.004977226739860008) q[4];
rz(-0.10158103627980354) q[4];
ry(0.6417781838237557) q[5];
rz(-2.509878363909359) q[5];
ry(-2.5151128828672356) q[6];
rz(-0.07946231524708477) q[6];
ry(-0.03753803969891867) q[7];
rz(-2.5122258822313728) q[7];
ry(-1.9774955931577036) q[8];
rz(-2.816986165680153) q[8];
ry(-1.2817873961069568) q[9];
rz(-2.504406534360867) q[9];
ry(3.126139259141308) q[10];
rz(2.86272024904793) q[10];
ry(2.0263007828747726) q[11];
rz(-0.37716807974556055) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.4028818453422289) q[0];
rz(0.12531870708816803) q[0];
ry(0.9851041704770328) q[1];
rz(0.5514335016454321) q[1];
ry(-2.614011927580358) q[2];
rz(3.1170846601591204) q[2];
ry(-1.486112453282608) q[3];
rz(-1.1483643617380164) q[3];
ry(1.2217623016363248) q[4];
rz(0.807890489850851) q[4];
ry(-2.7836563962734164) q[5];
rz(-1.2183316108788453) q[5];
ry(-2.1586831715829984) q[6];
rz(0.02248789574128971) q[6];
ry(-2.0127541530805706) q[7];
rz(-0.17642815769017428) q[7];
ry(-2.9908109405862486) q[8];
rz(-1.709968641793929) q[8];
ry(-1.5703002160637403) q[9];
rz(-0.5594464037389013) q[9];
ry(-0.10182742839975133) q[10];
rz(-1.3552548659677113) q[10];
ry(-2.0489683036008675) q[11];
rz(-2.8420000449442466) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.9681539750205106) q[0];
rz(1.6543347314349246) q[0];
ry(0.2624839087999318) q[1];
rz(1.647564422418088) q[1];
ry(3.1381639058931508) q[2];
rz(2.60384867317836) q[2];
ry(-3.1169193785657274) q[3];
rz(1.6560007402980041) q[3];
ry(0.004010938066234641) q[4];
rz(-3.0910719290862594) q[4];
ry(-3.1321789484437472) q[5];
rz(2.117309827300855) q[5];
ry(3.0849672323088666) q[6];
rz(-1.4113968431663801) q[6];
ry(3.045724240621127) q[7];
rz(1.1084057630677266) q[7];
ry(3.1352854992770474) q[8];
rz(-2.693182246535197) q[8];
ry(-0.0014354969068213208) q[9];
rz(-0.8608654117430549) q[9];
ry(1.5741236676765253) q[10];
rz(0.1287785114041866) q[10];
ry(-2.4226158298409244) q[11];
rz(-0.9545872533897616) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.7542939997383487) q[0];
rz(-1.4952768319467866) q[0];
ry(0.4641121899981165) q[1];
rz(-1.8995747203574236) q[1];
ry(-0.37777978563171016) q[2];
rz(-1.67277227086324) q[2];
ry(-0.02735646901388744) q[3];
rz(-0.2610278558290489) q[3];
ry(-1.8645851753904834) q[4];
rz(-2.643489659641466) q[4];
ry(2.987146715269796) q[5];
rz(-1.1903282078687165) q[5];
ry(1.1846335947560873) q[6];
rz(-1.333381442620424) q[6];
ry(1.9662321987856988) q[7];
rz(1.3914093324279868) q[7];
ry(-0.6040061116100431) q[8];
rz(-1.5292857449495845) q[8];
ry(0.10771291304919117) q[9];
rz(1.6036592988402958) q[9];
ry(-0.7575810325729314) q[10];
rz(0.08471366349163775) q[10];
ry(-1.477532659395543) q[11];
rz(1.7515240218658603) q[11];