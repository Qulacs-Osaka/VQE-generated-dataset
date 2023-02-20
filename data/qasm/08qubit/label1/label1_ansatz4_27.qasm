OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.144860167316775) q[0];
rz(1.4474376921670933) q[0];
ry(-1.5711820093433655) q[1];
rz(-0.43255928361478985) q[1];
ry(1.5771252938320686) q[2];
rz(1.5242282479831168) q[2];
ry(0.1110321789178883) q[3];
rz(1.7587967483013507) q[3];
ry(-0.3896368371282234) q[4];
rz(0.09105703944557526) q[4];
ry(1.6595689814087118) q[5];
rz(1.1467765762834663) q[5];
ry(-2.147713429995517) q[6];
rz(2.0789320302138576) q[6];
ry(2.633201978622482) q[7];
rz(-2.62100502850546) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4652655171880327) q[0];
rz(-0.9016221003637127) q[0];
ry(2.0412848330122126) q[1];
rz(-0.12154104508673531) q[1];
ry(2.1492649497455814) q[2];
rz(0.22618487405494037) q[2];
ry(-2.1046035747606) q[3];
rz(1.137978586084373) q[3];
ry(-1.6922422165674116) q[4];
rz(-1.3282095973020223) q[4];
ry(-1.8420248188251118) q[5];
rz(-0.37625807032952) q[5];
ry(-2.2932254694090717) q[6];
rz(-1.8307481227055113) q[6];
ry(2.252311085710877) q[7];
rz(1.0892586368786614) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.8038471693479385) q[0];
rz(1.3246598834328935) q[0];
ry(-1.9212606286282405) q[1];
rz(-1.1537363216654972) q[1];
ry(2.835658886510874) q[2];
rz(0.5737448096397513) q[2];
ry(-1.1127558657818375) q[3];
rz(1.3328582481010183) q[3];
ry(1.1676967299201562) q[4];
rz(-1.2683472674130227) q[4];
ry(-1.8536176465781797) q[5];
rz(-0.2772188701902998) q[5];
ry(-0.137247070517994) q[6];
rz(2.4426793258432267) q[6];
ry(-0.3657137823449364) q[7];
rz(1.900376581674789) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.966348045816475) q[0];
rz(-2.9667695497115214) q[0];
ry(-1.0028102778657253) q[1];
rz(0.5696415185210784) q[1];
ry(1.7436483658729087) q[2];
rz(-0.3246678896494907) q[2];
ry(0.6201117444504336) q[3];
rz(-2.96397692523341) q[3];
ry(3.032881998292716) q[4];
rz(1.5829481221828363) q[4];
ry(-1.224391343261714) q[5];
rz(1.2096671056263117) q[5];
ry(-1.4743015058086621) q[6];
rz(2.9418513282790535) q[6];
ry(1.4325790904960423) q[7];
rz(-0.7175805639576174) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.9639747149294884) q[0];
rz(0.7055060119961104) q[0];
ry(0.4868885544328938) q[1];
rz(0.2760106881902544) q[1];
ry(2.7065899211604916) q[2];
rz(-3.0989694520818016) q[2];
ry(-2.8039450213250072) q[3];
rz(0.32917813219785386) q[3];
ry(1.9231462169262916) q[4];
rz(1.0421027216162837) q[4];
ry(-2.7386712826841517) q[5];
rz(2.4092262864149214) q[5];
ry(-0.20246387421343215) q[6];
rz(2.0866170444505197) q[6];
ry(-0.8994167595115732) q[7];
rz(2.385611887221302) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.7197321574337519) q[0];
rz(2.1125468319677054) q[0];
ry(-1.255029344830768) q[1];
rz(2.933642458663136) q[1];
ry(2.5732258415492746) q[2];
rz(1.9313192102757313) q[2];
ry(2.9330738640139784) q[3];
rz(-1.1663813428238188) q[3];
ry(-1.0989018761473446) q[4];
rz(-2.3934013132173426) q[4];
ry(0.2469782873068173) q[5];
rz(1.4733142649931519) q[5];
ry(-1.8967681622483115) q[6];
rz(1.7991738811478142) q[6];
ry(2.3755911407652937) q[7];
rz(0.46728982435648875) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5176529627789466) q[0];
rz(1.936629768153965) q[0];
ry(1.7077702916872506) q[1];
rz(-2.679336837622982) q[1];
ry(-1.6186677897316284) q[2];
rz(3.00082696642798) q[2];
ry(-1.2041479907022712) q[3];
rz(-0.7076789234080101) q[3];
ry(2.083591075597943) q[4];
rz(2.077611873402942) q[4];
ry(-2.0873508820514406) q[5];
rz(1.699701929122826) q[5];
ry(-2.972398277196262) q[6];
rz(2.9812011655118433) q[6];
ry(2.2370139350223663) q[7];
rz(1.4322123293020377) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.23434902056406504) q[0];
rz(2.335847294281215) q[0];
ry(0.15247906923057922) q[1];
rz(2.8145992956303245) q[1];
ry(-2.1239602305603116) q[2];
rz(-2.799717659276551) q[2];
ry(-2.9606539359856225) q[3];
rz(1.6665661038537447) q[3];
ry(-2.9382321297328704) q[4];
rz(0.6767787819040753) q[4];
ry(-1.1039098440959567) q[5];
rz(-2.855488565481795) q[5];
ry(2.8157793606318844) q[6];
rz(-2.4576620901202646) q[6];
ry(2.1440213320251464) q[7];
rz(-2.081312613766956) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.2767275656652279) q[0];
rz(-2.409640527339175) q[0];
ry(1.437007261440482) q[1];
rz(2.345949844220276) q[1];
ry(-2.2954703929982467) q[2];
rz(3.0029317598249174) q[2];
ry(0.006715690415767028) q[3];
rz(-2.3081869344603474) q[3];
ry(-2.7263678966836067) q[4];
rz(2.809135084822181) q[4];
ry(-3.0725272967388784) q[5];
rz(-1.8119935748660398) q[5];
ry(-2.0844480234018543) q[6];
rz(1.3879561563901182) q[6];
ry(2.2642990963053737) q[7];
rz(0.8273432693859579) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.43419428526115583) q[0];
rz(2.9588263635699428) q[0];
ry(-2.681085242192135) q[1];
rz(1.6106587794359546) q[1];
ry(-1.6778135868350725) q[2];
rz(-0.7932375365065383) q[2];
ry(-0.9775010784587558) q[3];
rz(2.482910079370446) q[3];
ry(-0.194253699187982) q[4];
rz(1.7993021115970707) q[4];
ry(2.8702547199569612) q[5];
rz(-2.1499702295662337) q[5];
ry(-2.821828670266113) q[6];
rz(2.8610527554770178) q[6];
ry(0.9765584251886441) q[7];
rz(2.8399815130971975) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.1115601952973158) q[0];
rz(1.9002609755677549) q[0];
ry(1.6613019371603093) q[1];
rz(1.7329993750205857) q[1];
ry(-0.17730267474112585) q[2];
rz(-1.8601786053362606) q[2];
ry(-0.5407885767410701) q[3];
rz(-2.8446073587821443) q[3];
ry(1.9195932747069127) q[4];
rz(-2.8227403291470123) q[4];
ry(0.6979328235055249) q[5];
rz(1.577893381494083) q[5];
ry(2.4489905593774086) q[6];
rz(-2.540062152401581) q[6];
ry(1.541066999604024) q[7];
rz(-2.6085588799069384) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.9904304002126074) q[0];
rz(2.632133473874834) q[0];
ry(1.1573595046224767) q[1];
rz(1.8562860441192381) q[1];
ry(2.4013619086480738) q[2];
rz(-0.9566659692571368) q[2];
ry(-0.8812419790946375) q[3];
rz(-0.8822794139746212) q[3];
ry(-0.2849031160105395) q[4];
rz(0.13723351292067057) q[4];
ry(-0.37352020635665095) q[5];
rz(2.5756655086599762) q[5];
ry(-0.5245725022763744) q[6];
rz(-0.2233267875720708) q[6];
ry(2.418372883803793) q[7];
rz(1.7023038097074787) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.7349542735024714) q[0];
rz(1.7748060355204065) q[0];
ry(-1.6318554145555593) q[1];
rz(2.9841770184574323) q[1];
ry(0.2702240385683081) q[2];
rz(-0.9464010584305891) q[2];
ry(-2.9515914222196216) q[3];
rz(-2.3497486842100583) q[3];
ry(-0.46610804746531187) q[4];
rz(-2.4709738418183833) q[4];
ry(0.4664918345613064) q[5];
rz(2.795580916157274) q[5];
ry(-0.45236788180357923) q[6];
rz(-0.2993996427479566) q[6];
ry(-2.9522640032956993) q[7];
rz(-2.2022754026864595) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.0179089572939013) q[0];
rz(-2.1430974757733114) q[0];
ry(2.4952171072319085) q[1];
rz(3.1109289680803967) q[1];
ry(-1.9888547488753163) q[2];
rz(-1.9549991442740842) q[2];
ry(2.877762193040629) q[3];
rz(-2.731768981768901) q[3];
ry(0.5888258940496076) q[4];
rz(-2.7596479059839054) q[4];
ry(0.47138677349585134) q[5];
rz(2.4686966999156725) q[5];
ry(0.03927734825411644) q[6];
rz(-1.1003290539593902) q[6];
ry(-1.130209376721005) q[7];
rz(-2.5192684029464107) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1482003340576847) q[0];
rz(0.7272767286685654) q[0];
ry(2.8788205955117796) q[1];
rz(-2.714944494395835) q[1];
ry(-0.7709164490110288) q[2];
rz(-1.9675429784455574) q[2];
ry(-1.4375660246637272) q[3];
rz(1.951260907642606) q[3];
ry(-1.8191209244840412) q[4];
rz(0.12339406356400141) q[4];
ry(-0.6818785345930207) q[5];
rz(1.5200040489313347) q[5];
ry(3.0450869439219272) q[6];
rz(-0.9566018134648129) q[6];
ry(-2.531913011886204) q[7];
rz(-0.5755140412426185) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.7196231892348477) q[0];
rz(-0.9772790733015561) q[0];
ry(-0.2721382669245752) q[1];
rz(0.05344613573454194) q[1];
ry(2.9739843383216367) q[2];
rz(2.282117191299382) q[2];
ry(2.0451822609913615) q[3];
rz(-0.34754798287413274) q[3];
ry(-0.6728807291186302) q[4];
rz(3.1079852486764246) q[4];
ry(1.412189302425112) q[5];
rz(1.008166537102583) q[5];
ry(1.2879502617186063) q[6];
rz(-0.6406853344348853) q[6];
ry(-0.9429526220116486) q[7];
rz(-2.1770045467548638) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.3340424629524872) q[0];
rz(2.8175520180401445) q[0];
ry(-2.1420524264503467) q[1];
rz(2.5595049810919273) q[1];
ry(-1.9940523121292009) q[2];
rz(0.702303503472729) q[2];
ry(-3.0754171373336856) q[3];
rz(0.632709709278144) q[3];
ry(1.2937093673919238) q[4];
rz(-1.2526932217718298) q[4];
ry(0.20753614023371283) q[5];
rz(1.5329615284741225) q[5];
ry(-1.2260772431346914) q[6];
rz(-2.4790396099341283) q[6];
ry(1.2936850299912948) q[7];
rz(-1.513391678434326) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.913314716986338) q[0];
rz(1.9108164793383597) q[0];
ry(-2.8988031943793553) q[1];
rz(-1.1005699840382057) q[1];
ry(0.39260234655281306) q[2];
rz(-3.070503130391537) q[2];
ry(0.7031603433377134) q[3];
rz(2.5302317463768995) q[3];
ry(-2.39319505981586) q[4];
rz(0.6137197905074121) q[4];
ry(-0.2267052481931726) q[5];
rz(-0.0879445889616699) q[5];
ry(-0.18795683112437975) q[6];
rz(2.674339051076972) q[6];
ry(-2.6295550842557147) q[7];
rz(-0.8519355829038132) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9393474174131643) q[0];
rz(-2.5579038036295847) q[0];
ry(-0.3157765147295181) q[1];
rz(-2.9312596512663136) q[1];
ry(-0.5257661310901546) q[2];
rz(-0.8384466471075172) q[2];
ry(-1.1464465841985627) q[3];
rz(1.9092499944123955) q[3];
ry(-1.9448643151091358) q[4];
rz(2.4043890305609903) q[4];
ry(-0.3306201189229192) q[5];
rz(1.2783337467398763) q[5];
ry(1.6035715889545066) q[6];
rz(-1.1108662278070618) q[6];
ry(-1.4282034876990772) q[7];
rz(-1.6397207837116865) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.8189097630196773) q[0];
rz(-0.008769220529734945) q[0];
ry(-0.32402535839653074) q[1];
rz(2.838805650059225) q[1];
ry(0.5087146646866989) q[2];
rz(3.1047500633582295) q[2];
ry(-0.9800329656530583) q[3];
rz(-2.2899099458092538) q[3];
ry(1.2881723356753998) q[4];
rz(-1.3352810084495823) q[4];
ry(-1.6129463868035776) q[5];
rz(0.0574568746256121) q[5];
ry(-1.3082014451231532) q[6];
rz(-0.7392911084116802) q[6];
ry(-2.4570054191966864) q[7];
rz(0.16005233509437017) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.423157554814179) q[0];
rz(-1.6434250251502256) q[0];
ry(-2.5640872192015602) q[1];
rz(-2.840925316693492) q[1];
ry(-0.5023378384437859) q[2];
rz(-1.403174645628339) q[2];
ry(1.930531442990079) q[3];
rz(-3.0777180206839216) q[3];
ry(0.7755395261837039) q[4];
rz(3.0674292502379554) q[4];
ry(0.11816754453995199) q[5];
rz(2.474487571996145) q[5];
ry(-0.034247079039985834) q[6];
rz(0.5735788320240198) q[6];
ry(1.7671099192204844) q[7];
rz(-2.469938005621696) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.839391332016594) q[0];
rz(0.372441614559777) q[0];
ry(-1.1667189542514533) q[1];
rz(2.5967942827017154) q[1];
ry(1.3334675342824065) q[2];
rz(3.02406127805519) q[2];
ry(-1.7586831640123535) q[3];
rz(-2.027591024865166) q[3];
ry(-1.685847795409619) q[4];
rz(-2.8557414715434652) q[4];
ry(-1.3890237785647466) q[5];
rz(-2.0428855090971973) q[5];
ry(0.7205801311130826) q[6];
rz(2.0534898730858506) q[6];
ry(0.08654436127706688) q[7];
rz(-2.679149543265709) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.7382660063389763) q[0];
rz(0.4782103460415392) q[0];
ry(-0.8002975524804989) q[1];
rz(-0.4233142985308147) q[1];
ry(1.7019675620350254) q[2];
rz(-0.9956606749826182) q[2];
ry(-1.1332386500382368) q[3];
rz(1.3291793996898822) q[3];
ry(2.199810361080802) q[4];
rz(-2.2700910080477947) q[4];
ry(-0.5621143908492119) q[5];
rz(1.30400944095414) q[5];
ry(-0.07789856611580043) q[6];
rz(-2.2345285313998158) q[6];
ry(-2.055358934908795) q[7];
rz(1.3137907853201591) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.3791239753336955) q[0];
rz(2.6002317141978075) q[0];
ry(1.4185327147817315) q[1];
rz(-0.1113012989198694) q[1];
ry(1.3137121410935384) q[2];
rz(-2.121324252445054) q[2];
ry(0.0648421221542792) q[3];
rz(1.4823056804474453) q[3];
ry(1.8244019381899017) q[4];
rz(-1.9939038311622754) q[4];
ry(-1.5948653289527521) q[5];
rz(-2.5745348180304704) q[5];
ry(-2.393533289696479) q[6];
rz(-1.6213387466628852) q[6];
ry(-1.9131277281130612) q[7];
rz(-0.0015973575397065787) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6545936029103996) q[0];
rz(-2.1390740834717947) q[0];
ry(0.6146368411165462) q[1];
rz(2.4834936982753804) q[1];
ry(-0.791261492814031) q[2];
rz(-1.9170143202690657) q[2];
ry(-2.0094481747498447) q[3];
rz(2.1576268158690306) q[3];
ry(1.3019206303150432) q[4];
rz(-0.1571746303514914) q[4];
ry(1.3916488150889759) q[5];
rz(-0.6851101770576689) q[5];
ry(0.4058354656289927) q[6];
rz(0.8497716471503434) q[6];
ry(0.3435321749981126) q[7];
rz(1.239689298294544) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6769787126815245) q[0];
rz(2.9947008359020866) q[0];
ry(2.9809742219561746) q[1];
rz(0.7808697583058885) q[1];
ry(0.938972581069537) q[2];
rz(0.5320378481162327) q[2];
ry(-2.428915242590157) q[3];
rz(0.5425396383839846) q[3];
ry(1.4885404845585555) q[4];
rz(-1.93829098229059) q[4];
ry(1.4180152711145788) q[5];
rz(2.5184538016516007) q[5];
ry(2.2780256552944183) q[6];
rz(2.1006590246421024) q[6];
ry(1.4580614871457502) q[7];
rz(-2.8946731845241325) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.0977951559436896) q[0];
rz(0.1900554456991577) q[0];
ry(-2.7972937924057484) q[1];
rz(-0.4890728956209669) q[1];
ry(-0.1529650948818642) q[2];
rz(-2.6269280132785906) q[2];
ry(2.2340586611125888) q[3];
rz(2.296201191178667) q[3];
ry(1.9759872002036538) q[4];
rz(-2.078827293402515) q[4];
ry(0.22081278323954656) q[5];
rz(1.0504696220472836) q[5];
ry(2.104568893765529) q[6];
rz(-2.911225690272638) q[6];
ry(2.2398287133844033) q[7];
rz(-0.23118962217889039) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.462374197506428) q[0];
rz(-1.067463077672321) q[0];
ry(1.8279922078850546) q[1];
rz(-1.5126156508475084) q[1];
ry(1.9731272516776395) q[2];
rz(-1.6609423226042113) q[2];
ry(-0.34285652631328745) q[3];
rz(-1.12715851403485) q[3];
ry(-1.0420756338247594) q[4];
rz(-0.12474740024828802) q[4];
ry(-2.9963109934234176) q[5];
rz(1.6335189053077983) q[5];
ry(-2.4709000857829477) q[6];
rz(2.8625484654742084) q[6];
ry(-1.4558423066616255) q[7];
rz(-2.2371666770291228) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.9643798610205705) q[0];
rz(1.267470971894972) q[0];
ry(-2.9011086722016497) q[1];
rz(1.9759094981647385) q[1];
ry(-1.7665321412889732) q[2];
rz(-1.1284967071632952) q[2];
ry(0.7693815294633085) q[3];
rz(2.777122110483531) q[3];
ry(-2.8170627025334145) q[4];
rz(0.4428474455271641) q[4];
ry(0.27418863707331953) q[5];
rz(-0.8941727857994516) q[5];
ry(1.9533136478157378) q[6];
rz(-0.7632017552693925) q[6];
ry(2.883427383470606) q[7];
rz(-2.6810617277854876) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.049903502410833) q[0];
rz(2.2676729829034166) q[0];
ry(-2.473272749048275) q[1];
rz(2.9409437321749365) q[1];
ry(2.562779222432901) q[2];
rz(-2.3177375918793004) q[2];
ry(-2.639952267782057) q[3];
rz(2.8083743420599463) q[3];
ry(1.0008070786351055) q[4];
rz(1.901570960394156) q[4];
ry(0.04424710906258082) q[5];
rz(-0.41574668318117386) q[5];
ry(-1.0386580323885326) q[6];
rz(-2.514788723998955) q[6];
ry(-0.4315246343898123) q[7];
rz(-0.8241074533901313) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.270636968808856) q[0];
rz(0.41323042363949347) q[0];
ry(1.5322077860874288) q[1];
rz(0.9783526060989924) q[1];
ry(-2.375855330873695) q[2];
rz(1.0651325214332885) q[2];
ry(0.24117209141852095) q[3];
rz(1.612512906236926) q[3];
ry(1.3173840418485376) q[4];
rz(1.962266310777384) q[4];
ry(-2.5222774540400548) q[5];
rz(-2.2788759323888295) q[5];
ry(-1.6870172923891307) q[6];
rz(2.292005449238593) q[6];
ry(2.2267878649354778) q[7];
rz(-1.3060613876387297) q[7];