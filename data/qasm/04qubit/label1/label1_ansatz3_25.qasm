OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.3340617891137034) q[0];
rz(-2.6499481671248546) q[0];
ry(-2.4105677981477376) q[1];
rz(0.8634017857301094) q[1];
ry(2.674119000878704) q[2];
rz(2.157252900569673) q[2];
ry(-2.622031061624988) q[3];
rz(2.1331468232971154) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.9693926340591954) q[0];
rz(-0.39308108009053017) q[0];
ry(3.0251198543956956) q[1];
rz(0.19733228171080341) q[1];
ry(0.37033057911836714) q[2];
rz(-0.18439742750807042) q[2];
ry(0.5950141422015411) q[3];
rz(-2.3982913501971233) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9206052511954822) q[0];
rz(-0.9189490447779313) q[0];
ry(-1.0543073718423026) q[1];
rz(-1.2409622396681428) q[1];
ry(-1.5869294893867083) q[2];
rz(-1.5777406924589499) q[2];
ry(0.5699923729681284) q[3];
rz(1.3912200219018205) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.6538491434980462) q[0];
rz(2.4393724316196277) q[0];
ry(2.13903930936165) q[1];
rz(-2.3073223284859514) q[1];
ry(0.6029884771753933) q[2];
rz(-1.2598613864537906) q[2];
ry(-0.6456713513708634) q[3];
rz(-0.1498146620466354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3820316369119487) q[0];
rz(1.5779512433973408) q[0];
ry(1.0280905443660178) q[1];
rz(0.05165170375655363) q[1];
ry(-2.3699991083964944) q[2];
rz(-1.0430042981483008) q[2];
ry(2.4124640305438927) q[3];
rz(0.19370217843662046) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.6417231669860572) q[0];
rz(1.148415387601755) q[0];
ry(-0.48776234817513514) q[1];
rz(0.8036325927093888) q[1];
ry(-0.14260010154628855) q[2];
rz(-2.667875654520251) q[2];
ry(1.0189842187807567) q[3];
rz(-1.6445420034056275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9814571349338408) q[0];
rz(0.9607168091778842) q[0];
ry(1.9523356856829532) q[1];
rz(0.14476544807267014) q[1];
ry(-1.512110245320739) q[2];
rz(-1.6095868617798506) q[2];
ry(1.0209003311476677) q[3];
rz(-2.994101280100094) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.09589662571079756) q[0];
rz(-2.3285596420229435) q[0];
ry(0.09868939724401789) q[1];
rz(0.6304638385996286) q[1];
ry(-0.026495550088282402) q[2];
rz(0.09733933673587931) q[2];
ry(-1.4759372142699405) q[3];
rz(-0.7327746527880564) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.2251066083982507) q[0];
rz(-1.4285286242990993) q[0];
ry(2.437761068933387) q[1];
rz(-1.8381409261825379) q[1];
ry(0.03401547325131879) q[2];
rz(0.349462514555435) q[2];
ry(2.6363953357705716) q[3];
rz(2.976895166408765) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.8108709926797197) q[0];
rz(-1.0175238140049023) q[0];
ry(-1.9939984782335516) q[1];
rz(-0.8856604468419407) q[1];
ry(1.6678231921914455) q[2];
rz(-2.470042575525218) q[2];
ry(0.8317297574505345) q[3];
rz(-2.014629835644565) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.034171814404288625) q[0];
rz(1.9277987964493757) q[0];
ry(1.444002195725366) q[1];
rz(-2.258069089478682) q[1];
ry(-0.5053816151211779) q[2];
rz(-2.158944118549024) q[2];
ry(-0.27195966203340305) q[3];
rz(-2.3604422246257646) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.675332600631438) q[0];
rz(-0.42733109478023223) q[0];
ry(-0.3335830517550041) q[1];
rz(2.57329923918516) q[1];
ry(1.1620124023322127) q[2];
rz(1.1482015354861312) q[2];
ry(-2.434152531595926) q[3];
rz(-0.231281520990475) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.1518504383949777) q[0];
rz(1.8055192157856808) q[0];
ry(-1.0644857228977447) q[1];
rz(-0.16092752542977795) q[1];
ry(-2.7565080856448354) q[2];
rz(-0.01882791203775751) q[2];
ry(2.4175136320081356) q[3];
rz(-0.1488860289381435) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.2994806335810063) q[0];
rz(1.8923864977912812) q[0];
ry(2.950124609362776) q[1];
rz(2.304560421148785) q[1];
ry(1.44212269941388) q[2];
rz(-1.4369849151628717) q[2];
ry(-1.9099237915386675) q[3];
rz(3.112530330956293) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.22626822179566175) q[0];
rz(-2.0501028782968405) q[0];
ry(-0.8263805174485702) q[1];
rz(-2.1518148476079375) q[1];
ry(2.165509456849733) q[2];
rz(-0.15593454447177432) q[2];
ry(1.6173481618771444) q[3];
rz(0.001157763616611866) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6828196583227317) q[0];
rz(0.5116626031800207) q[0];
ry(-0.32523206776378455) q[1];
rz(2.926020307251285) q[1];
ry(0.9316899634545259) q[2];
rz(0.8085787441808225) q[2];
ry(-0.5941947620148875) q[3];
rz(-0.32918551450632233) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.2004803874626422) q[0];
rz(0.07323787511784818) q[0];
ry(0.7070364539419722) q[1];
rz(2.6873009924780753) q[1];
ry(1.3769695839446792) q[2];
rz(3.0427950354517392) q[2];
ry(-2.8630784398673956) q[3];
rz(1.5796878574065454) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.5494288967120635) q[0];
rz(-0.8664505709325324) q[0];
ry(2.614879046498095) q[1];
rz(2.339259982474878) q[1];
ry(-1.9712426183447513) q[2];
rz(-1.6437129621138031) q[2];
ry(-1.0082518440908634) q[3];
rz(2.385070218149639) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3508580066104308) q[0];
rz(1.6248676821024874) q[0];
ry(-1.9070995902396364) q[1];
rz(-1.7209352076101876) q[1];
ry(-0.9949813472254336) q[2];
rz(-1.9856335940086098) q[2];
ry(1.0991038794848862) q[3];
rz(2.539703559643755) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.1248547561352658) q[0];
rz(0.4672467988636102) q[0];
ry(-1.8583271853713101) q[1];
rz(2.578524752039744) q[1];
ry(-0.49135048760340516) q[2];
rz(1.2943977464081784) q[2];
ry(0.2112848057946186) q[3];
rz(1.7907143243640355) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.8066198911119438) q[0];
rz(-2.601151905792413) q[0];
ry(-0.4906275349401037) q[1];
rz(-2.433658234063916) q[1];
ry(-0.8510889274204958) q[2];
rz(-0.44553358482887956) q[2];
ry(-0.8837796936324827) q[3];
rz(-0.7751983862169103) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.18741863170921746) q[0];
rz(0.7318514784817148) q[0];
ry(2.6089553561743952) q[1];
rz(-1.4237434643296476) q[1];
ry(-2.2589608860297727) q[2];
rz(-1.6393948645349734) q[2];
ry(-2.81461895667242) q[3];
rz(-1.8834588371147332) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6873967912900774) q[0];
rz(2.624578140546438) q[0];
ry(-0.31370978977894254) q[1];
rz(-1.7551939489418027) q[1];
ry(-2.25118275910682) q[2];
rz(-1.8490234563980463) q[2];
ry(0.4085611326353342) q[3];
rz(-1.6294730659477257) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.6210581294475377) q[0];
rz(-0.21012231400012468) q[0];
ry(-0.5718915364767864) q[1];
rz(-1.9847560164507874) q[1];
ry(-1.3995742850912212) q[2];
rz(1.0189376728346724) q[2];
ry(-2.570111838580371) q[3];
rz(-3.1204114897312234) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.8448418096875446) q[0];
rz(2.815688228697695) q[0];
ry(0.6972729181512539) q[1];
rz(1.5458326649149532) q[1];
ry(2.3850243983200694) q[2];
rz(0.11938074207981143) q[2];
ry(-1.1922848365065866) q[3];
rz(0.826802167368272) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.8620848044553736) q[0];
rz(-1.3845647933964906) q[0];
ry(2.533812204216186) q[1];
rz(2.531408289870521) q[1];
ry(1.0356200546128218) q[2];
rz(-2.126890597477539) q[2];
ry(-1.9174934502821483) q[3];
rz(-0.21417995268152132) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.9589627505964664) q[0];
rz(1.6582480441395808) q[0];
ry(-2.505828434419385) q[1];
rz(2.6182644753181084) q[1];
ry(0.3565501340479358) q[2];
rz(0.9927597758155545) q[2];
ry(0.2699197844244469) q[3];
rz(2.643815961280883) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.537212449013758) q[0];
rz(0.24032061408649152) q[0];
ry(-1.1567849798083707) q[1];
rz(-1.0221253377551998) q[1];
ry(-0.219048712689073) q[2];
rz(-2.0850567194853173) q[2];
ry(1.5294433694664487) q[3];
rz(-0.7571840745334997) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.7666048409263713) q[0];
rz(-1.7702069913588412) q[0];
ry(-2.966996028337126) q[1];
rz(0.24283212220434428) q[1];
ry(3.0542527331061193) q[2];
rz(1.2689994254430914) q[2];
ry(1.2683112186067333) q[3];
rz(-2.4826364195640815) q[3];