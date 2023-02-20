OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.00014493579374164834) q[0];
rz(-2.8235303390259565) q[0];
ry(2.6664618637185415) q[1];
rz(0.6471746996656441) q[1];
ry(-1.8428545687615348) q[2];
rz(1.2779359950605258) q[2];
ry(3.1415124444089937) q[3];
rz(1.4863726274765305) q[3];
ry(-1.8104798177637669) q[4];
rz(-0.39015009270924494) q[4];
ry(-2.5391425206749374) q[5];
rz(-3.124461593300502) q[5];
ry(-1.9796400330543564) q[6];
rz(-2.124447230627513) q[6];
ry(0.08231370860837144) q[7];
rz(3.102256876173567) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(3.1414838440821597) q[0];
rz(-2.4539953926943867) q[0];
ry(2.5194629701868485) q[1];
rz(-1.4273060884857565) q[1];
ry(2.9802591009368298) q[2];
rz(0.19436177621000894) q[2];
ry(-0.00013462031738153968) q[3];
rz(-1.0405546904345595) q[3];
ry(3.1407386297721827) q[4];
rz(1.5802405990771913) q[4];
ry(-1.8922521781182116) q[5];
rz(0.12434579393923871) q[5];
ry(3.1401205064067965) q[6];
rz(-2.1263058677785542) q[6];
ry(1.595688418105909) q[7];
rz(2.3018152809206365) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.0001121613504485836) q[0];
rz(-0.8724655199295138) q[0];
ry(-3.062583127522143) q[1];
rz(1.515614477573299) q[1];
ry(-2.5538304900469964) q[2];
rz(-2.948119693986014) q[2];
ry(8.561044433985643e-05) q[3];
rz(-0.4169171683854364) q[3];
ry(-2.190559216921635) q[4];
rz(1.147383724739254) q[4];
ry(-3.0521016189073205) q[5];
rz(1.0771036222118915) q[5];
ry(-1.6335030021548498) q[6];
rz(-1.293540072728668) q[6];
ry(0.0010408133448178705) q[7];
rz(-0.3465212751828375) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5709077095848107) q[0];
rz(-2.1072516916906303) q[0];
ry(-0.2051640446646125) q[1];
rz(-1.4401982485015354) q[1];
ry(0.7723965124285135) q[2];
rz(-1.130514592418618) q[2];
ry(3.1414175192232627) q[3];
rz(3.1099583705712073) q[3];
ry(1.564564270043484) q[4];
rz(-1.5606098279570606) q[4];
ry(1.245743046778893) q[5];
rz(2.617694879294659) q[5];
ry(3.128045991074866) q[6];
rz(2.3614520914868864) q[6];
ry(2.439747444870072) q[7];
rz(-1.587298978590324) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-9.462049708598547e-06) q[0];
rz(-2.732767637590156) q[0];
ry(1.570325100404979) q[1];
rz(-0.8612755442350446) q[1];
ry(-3.1415774192212127) q[2];
rz(2.0122549413078072) q[2];
ry(-2.456206651461821e-06) q[3];
rz(1.5948220979584713) q[3];
ry(1.5708683113443467) q[4];
rz(-2.965755283169771) q[4];
ry(3.0799946334141004) q[5];
rz(-0.011911774260446846) q[5];
ry(2.938434757999939) q[6];
rz(1.495954759816575) q[6];
ry(3.1326875527446747) q[7];
rz(1.92304187338924) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.2762049279941774) q[0];
rz(-2.95640008863901) q[0];
ry(-0.6570145005650709) q[1];
rz(-3.050848018193763) q[1];
ry(-1.5710878646348532) q[2];
rz(0.3403243026938165) q[2];
ry(2.9340779317827534) q[3];
rz(-2.7278669425855973) q[3];
ry(-0.45941065807816267) q[4];
rz(-0.5573683516255423) q[4];
ry(-1.7809116061356614) q[5];
rz(-3.133940324998598) q[5];
ry(-1.0873321422044095) q[6];
rz(2.4543051128289384) q[6];
ry(-0.9570252887015901) q[7];
rz(2.866123096296922) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(3.139293433996258) q[0];
rz(-0.3654611176876575) q[0];
ry(3.1411133431184703) q[1];
rz(-2.909168676890537) q[1];
ry(-9.542971088905006e-07) q[2];
rz(-2.2319644507862786) q[2];
ry(-3.141419329389827) q[3];
rz(0.4138038085469457) q[3];
ry(-0.11972525892339103) q[4];
rz(0.7726820594734288) q[4];
ry(1.5793116702960734) q[5];
rz(-3.140333724490887) q[5];
ry(-3.073680215810275) q[6];
rz(-2.2723964355554873) q[6];
ry(-0.1411179156691497) q[7];
rz(0.5044048052667477) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.9702058261868373) q[0];
rz(1.865640212912285) q[0];
ry(0.6913321893300162) q[1];
rz(2.144892833060333) q[1];
ry(-0.3389610164538425) q[2];
rz(1.6012258520688543) q[2];
ry(-1.572595800252795) q[3];
rz(-2.7362624125439687) q[3];
ry(-0.0005848378581907454) q[4];
rz(-0.9354971027017341) q[4];
ry(-1.786080472212185) q[5];
rz(-1.5843986276609536) q[5];
ry(-1.5259803237071397) q[6];
rz(-0.6874868458489374) q[6];
ry(4.069478716317576e-05) q[7];
rz(-2.960903327756002) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.9715798480943295e-05) q[0];
rz(-3.046229345063214) q[0];
ry(1.5870510426388977) q[1];
rz(1.2416571818136848) q[1];
ry(-0.00010984057732521536) q[2];
rz(-1.6004213933116915) q[2];
ry(3.1415710051133687) q[3];
rz(2.168994318425709) q[3];
ry(-0.0022781389086459214) q[4];
rz(-2.6564312638325727) q[4];
ry(-1.573652214497658) q[5];
rz(1.787327790989829) q[5];
ry(-2.9199700820010133) q[6];
rz(1.5207752024669763) q[6];
ry(-1.4253505616807125) q[7];
rz(0.0313203839816989) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.6212613871307566) q[0];
rz(-3.0345019613241253) q[0];
ry(-3.1054722073403553) q[1];
rz(1.2572283998801914) q[1];
ry(2.8028504900014597) q[2];
rz(1.9511581327367473) q[2];
ry(3.141578263437578) q[3];
rz(-2.5271297007117117) q[3];
ry(1.5707813948344305) q[4];
rz(-1.2536934364641352) q[4];
ry(-3.1291106824207326) q[5];
rz(-2.5643220562131432) q[5];
ry(2.387068025519526) q[6];
rz(1.2386106247686646) q[6];
ry(-3.1415738884274886) q[7];
rz(-1.6383800988280104) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.572503323634299) q[0];
rz(-1.5699508195885095) q[0];
ry(-2.9465157868074243) q[1];
rz(-0.9059750081540959) q[1];
ry(1.5572219477322102) q[2];
rz(-1.3049371874644144) q[2];
ry(3.141586352590856) q[3];
rz(0.6333775257238021) q[3];
ry(0.00014197732120871864) q[4];
rz(-0.5887544188398337) q[4];
ry(-1.5215641354776575) q[5];
rz(1.6470852662462498) q[5];
ry(-2.3163344913922224) q[6];
rz(3.140571361362285) q[6];
ry(0.04120052505187103) q[7];
rz(0.9149906983467115) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.563765671852873) q[0];
rz(0.7807615653482098) q[0];
ry(1.5706581240435842) q[1];
rz(1.8664119454409178) q[1];
ry(-7.98428830428577e-06) q[2];
rz(-3.105862337146716) q[2];
ry(-1.2537671511658743e-05) q[3];
rz(-1.9516554197309237) q[3];
ry(0.0001524709956353121) q[4];
rz(-1.571453550324048) q[4];
ry(-1.569595077575214) q[5];
rz(-2.167508647186703) q[5];
ry(1.2057140626227414) q[6];
rz(1.8097155037148036) q[6];
ry(-3.1415393694211105) q[7];
rz(2.930048776109236) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.3571927409130788) q[0];
rz(2.1788463274518035) q[0];
ry(2.3289020120529744) q[1];
rz(-2.327608707019288) q[1];
ry(-0.8249710670383644) q[2];
rz(0.3852165264194065) q[2];
ry(1.9708636312198555) q[3];
rz(-0.3514927585332872) q[3];
ry(0.8044715373282746) q[4];
rz(-1.9583326555018923) q[4];
ry(-1.0223770875905478) q[5];
rz(-1.0650978321935085) q[5];
ry(-3.0241811855840286) q[6];
rz(2.5008369949676297) q[6];
ry(0.8037452977985905) q[7];
rz(0.800510386106858) q[7];