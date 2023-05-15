OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.8215367322103921) q[0];
rz(-1.0487511110904686) q[0];
ry(-1.879366079682657) q[1];
rz(2.6772127424296888) q[1];
ry(3.1394113797504324) q[2];
rz(-2.295914326437793) q[2];
ry(0.6490270059947634) q[3];
rz(-0.5659105925282573) q[3];
ry(3.1412019416177372) q[4];
rz(-0.9394006014637748) q[4];
ry(-0.5277009063294589) q[5];
rz(-0.483881792406304) q[5];
ry(-3.139877115763491) q[6];
rz(-2.8186148232561936) q[6];
ry(1.965498808476016) q[7];
rz(2.486581954438904) q[7];
ry(-0.3631683985536318) q[8];
rz(-0.5814641657296695) q[8];
ry(0.6677714608155219) q[9];
rz(2.22475132949613) q[9];
ry(0.8494969587793286) q[10];
rz(1.5522894983884845) q[10];
ry(3.131188503012964) q[11];
rz(-1.9048274802099903) q[11];
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
ry(-2.517695593784836) q[0];
rz(-0.683193890188359) q[0];
ry(3.1304760257899518) q[1];
rz(-1.4693566717488862) q[1];
ry(2.245894962307917) q[2];
rz(-1.4190190240858902) q[2];
ry(-2.871592466919332) q[3];
rz(1.4651896436489613) q[3];
ry(-1.0800314647735307) q[4];
rz(-1.345371756383844) q[4];
ry(-1.0964141917019186) q[5];
rz(-1.3461739440987999) q[5];
ry(-0.003411645451139478) q[6];
rz(0.7019450182414838) q[6];
ry(-0.646032305268501) q[7];
rz(2.400982509493621) q[7];
ry(-1.1569284979185506) q[8];
rz(-0.11524096517156686) q[8];
ry(-2.6847580061226237) q[9];
rz(0.09423714050241498) q[9];
ry(0.6646191061651093) q[10];
rz(-1.6392827273417294) q[10];
ry(-0.008519499935422939) q[11];
rz(-1.0998811466326646) q[11];
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
ry(-0.9174837631306612) q[0];
rz(-2.256327528022517) q[0];
ry(-0.5010921586413729) q[1];
rz(0.3717428606214561) q[1];
ry(3.1392323405361116) q[2];
rz(1.8212056547153566) q[2];
ry(-3.1415322363009595) q[3];
rz(-1.1218602683693646) q[3];
ry(0.009853410277286478) q[4];
rz(0.5137598389641164) q[4];
ry(2.4758857026839505) q[5];
rz(0.05261239778334399) q[5];
ry(-1.5942047978615923) q[6];
rz(1.540427288113599) q[6];
ry(0.8694032873906457) q[7];
rz(1.382802083517264) q[7];
ry(-1.9744332419821369) q[8];
rz(2.6240216279030935) q[8];
ry(-2.4321938862347663) q[9];
rz(0.897703056923075) q[9];
ry(2.471311884210813) q[10];
rz(2.028574838707093) q[10];
ry(0.003267017718219023) q[11];
rz(2.740856749309045) q[11];
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
ry(-0.07174972717387362) q[0];
rz(0.32690638258949967) q[0];
ry(1.63936417577827) q[1];
rz(-2.0948738469217814) q[1];
ry(-0.7635284250765757) q[2];
rz(1.0278978014208322) q[2];
ry(0.7110130331984283) q[3];
rz(1.5405290681906063) q[3];
ry(-0.4959059712617502) q[4];
rz(-3.004641963573279) q[4];
ry(3.1356550997029426) q[5];
rz(0.05646769228104599) q[5];
ry(2.99208271606163) q[6];
rz(-2.2215793226903537) q[6];
ry(-2.516843356733296) q[7];
rz(2.7237341803920705) q[7];
ry(-2.288163826280633) q[8];
rz(-2.41071226186763) q[8];
ry(-1.674416835730458) q[9];
rz(1.957177490396439) q[9];
ry(-1.574562659149465) q[10];
rz(-3.009091072275264) q[10];
ry(-0.006392363058935758) q[11];
rz(2.5727041637093238) q[11];
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
ry(-2.8523411186394774) q[0];
rz(-0.22518009423806976) q[0];
ry(0.4058780225705401) q[1];
rz(-1.4602782418121967) q[1];
ry(0.001862605818416604) q[2];
rz(-3.06974609819484) q[2];
ry(0.4611704955953071) q[3];
rz(-0.004129337724428417) q[3];
ry(-1.1220129181532685) q[4];
rz(0.004784143439590771) q[4];
ry(-1.5673890566575441) q[5];
rz(1.5743542176227472) q[5];
ry(1.5594639853393586) q[6];
rz(-3.0965057867544155) q[6];
ry(-0.02247568027379981) q[7];
rz(1.9531992745136204) q[7];
ry(3.13888723064034) q[8];
rz(-0.5449303225206172) q[8];
ry(-0.3100003900306199) q[9];
rz(-0.4722354597935418) q[9];
ry(-1.8943363448907053) q[10];
rz(1.9455028712032902) q[10];
ry(1.6384950640020524) q[11];
rz(-1.5082216790096121) q[11];
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
ry(-3.1255942966227668) q[0];
rz(1.8704072745496187) q[0];
ry(2.586789167226284) q[1];
rz(2.251196797993853) q[1];
ry(1.56938445560988) q[2];
rz(-0.011145140717034523) q[2];
ry(0.6584889772256002) q[3];
rz(2.6164402989911695) q[3];
ry(1.5715634516818797) q[4];
rz(3.141228197113519) q[4];
ry(1.7420272324778034) q[5];
rz(0.004002732774680436) q[5];
ry(1.5718170095787027) q[6];
rz(1.8806792570712643) q[6];
ry(0.00125146697792265) q[7];
rz(-1.534151478569033) q[7];
ry(-1.2499761355589127) q[8];
rz(0.20752618425892494) q[8];
ry(1.6319940194795037) q[9];
rz(-0.30982280685369723) q[9];
ry(-1.547261506594925) q[10];
rz(1.999138202839471) q[10];
ry(1.3423536437855625) q[11];
rz(1.4014838981306754) q[11];
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
ry(1.452870813544041) q[0];
rz(2.7722075251073277) q[0];
ry(-1.5795721738898334) q[1];
rz(-1.8886869415134686) q[1];
ry(-1.6888123766812573) q[2];
rz(-3.140607186822653) q[2];
ry(2.481250709760746) q[3];
rz(2.8998182802274783) q[3];
ry(-1.6756842606765119) q[4];
rz(-3.1409718412335876) q[4];
ry(1.56888706681447) q[5];
rz(-0.00044588223510988456) q[5];
ry(-3.8160560339228766e-05) q[6];
rz(1.2603707801030222) q[6];
ry(-1.5722582807585912) q[7];
rz(-0.007641624471123122) q[7];
ry(3.137854209331819) q[8];
rz(-1.315725292355409) q[8];
ry(-1.5671344996404182) q[9];
rz(-1.770059035963465) q[9];
ry(-1.0263701853658584) q[10];
rz(-2.631039112817452) q[10];
ry(1.235417287731034) q[11];
rz(-0.39737201487843377) q[11];
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
ry(3.0374608387503756) q[0];
rz(1.1843602715677832) q[0];
ry(-1.8872394693877022) q[1];
rz(-2.6081945320919644) q[1];
ry(1.823288358692686) q[2];
rz(-3.1385799208361136) q[2];
ry(-0.27285755489150354) q[3];
rz(-3.1216873608578646) q[3];
ry(1.542791115498388) q[4];
rz(-1.1866119851537755) q[4];
ry(1.0265622516615602) q[5];
rz(-3.1354701659754283) q[5];
ry(-1.5818499051038755) q[6];
rz(-3.1412965231331818) q[6];
ry(1.5547383711829745) q[7];
rz(3.140014548253089) q[7];
ry(-1.5710450820158606) q[8];
rz(2.4317748883273933) q[8];
ry(-3.1395567736068317) q[9];
rz(2.9401173700852854) q[9];
ry(-3.135326402260267) q[10];
rz(-0.36666631239888225) q[10];
ry(3.0819400229817178) q[11];
rz(2.43680555566906) q[11];
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
ry(3.141245872785468) q[0];
rz(0.10149664481341512) q[0];
ry(3.110390722936973) q[1];
rz(-1.762009566383952) q[1];
ry(2.77492233303228) q[2];
rz(-3.1381837779986976) q[2];
ry(-0.6035621542268313) q[3];
rz(-3.081143550691963) q[3];
ry(-3.1402863346448004) q[4];
rz(-0.2117638590094657) q[4];
ry(-1.5760962776841705) q[5];
rz(-0.6422569119594553) q[5];
ry(1.5645367405880535) q[6];
rz(-0.00172421799291119) q[6];
ry(-2.5124532757338764) q[7];
rz(0.013465862138815508) q[7];
ry(-0.0012526152037173333) q[8];
rz(0.7117648805766583) q[8];
ry(-1.5721237621541535) q[9];
rz(-3.1404001172252554) q[9];
ry(2.416629809797712) q[10];
rz(-0.9130145646807246) q[10];
ry(-2.6198040516201075) q[11];
rz(2.8290020928425936) q[11];
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
ry(2.1911067486087448) q[0];
rz(-3.0219480195325055) q[0];
ry(-1.1041255582755038) q[1];
rz(1.5550709569312084) q[1];
ry(1.975689007085509) q[2];
rz(0.011771720721720946) q[2];
ry(1.7960951586706557) q[3];
rz(-1.8162525242974334) q[3];
ry(-0.7830916552246389) q[4];
rz(-0.744344964055224) q[4];
ry(-0.28015392565773417) q[5];
rz(1.1355299649975361) q[5];
ry(-1.5599947837392525) q[6];
rz(-0.00905322801283326) q[6];
ry(1.2926588386016726) q[7];
rz(0.16585430811145868) q[7];
ry(-1.5491024689922135) q[8];
rz(-3.140280992408308) q[8];
ry(-1.5122865005027204) q[9];
rz(0.26694135818605075) q[9];
ry(-1.570123598785311) q[10];
rz(5.390377371572441e-05) q[10];
ry(-3.0773076370006582) q[11];
rz(-0.01704173170818703) q[11];
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
ry(1.4837953476412469) q[0];
rz(0.0020600252774500495) q[0];
ry(1.576361879804832) q[1];
rz(-1.7453595022285866) q[1];
ry(-3.141358455064207) q[2];
rz(0.011751473192981088) q[2];
ry(2.7822379677466813e-05) q[3];
rz(0.6074251009360162) q[3];
ry(-3.138561077892594) q[4];
rz(-2.763670120022809) q[4];
ry(0.006854721510782955) q[5];
rz(2.3563683446861896) q[5];
ry(0.0008456928271064961) q[6];
rz(1.933981280717889) q[6];
ry(2.7189230219094527) q[7];
rz(0.3528321424907368) q[7];
ry(-0.36351505458909666) q[8];
rz(3.0656336461498777) q[8];
ry(-1.7517912264018454) q[9];
rz(-0.7586867707618685) q[9];
ry(-1.632412251415276) q[10];
rz(3.0513341759296235) q[10];
ry(-1.5982181725182247) q[11];
rz(-2.952998911377256) q[11];
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
ry(1.5919679392349355) q[0];
rz(3.141562449686081) q[0];
ry(-3.0857870936357155) q[1];
rz(-1.7577069547562612) q[1];
ry(1.557488638614477) q[2];
rz(0.03138706145722293) q[2];
ry(-3.0754569846059314) q[3];
rz(1.9062223657821316) q[3];
ry(-1.977253331084989) q[4];
rz(0.6851513150000301) q[4];
ry(0.5550196516841822) q[5];
rz(0.3189238574536098) q[5];
ry(0.0019006688953840166) q[6];
rz(-2.3729176362218336) q[6];
ry(2.806778035205335) q[7];
rz(2.960923091590686) q[7];
ry(-0.1385440040638987) q[8];
rz(2.7977023469105995) q[8];
ry(3.0848025621085453) q[9];
rz(1.367677605980746) q[9];
ry(-3.133150087633042) q[10];
rz(2.9289705918267726) q[10];
ry(-1.8778477989343956) q[11];
rz(-3.034615695742643) q[11];
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
ry(1.2305047078780325) q[0];
rz(-1.5736459359058594) q[0];
ry(1.548848601090376) q[1];
rz(-1.5710330703042796) q[1];
ry(0.051987413516907374) q[2];
rz(-1.602355908375121) q[2];
ry(-0.6145720487197224) q[3];
rz(-1.5566896482515817) q[3];
ry(2.6617986231313684) q[4];
rz(1.5667372540149407) q[4];
ry(1.576492317337283) q[5];
rz(1.5722706277958052) q[5];
ry(-3.137656350901546) q[6];
rz(1.1229417459010866) q[6];
ry(0.19303428990477123) q[7];
rz(1.9398904413332732) q[7];
ry(3.141269060443055) q[8];
rz(1.1532755351550303) q[8];
ry(-2.808131016024267) q[9];
rz(0.5803853446349193) q[9];
ry(-0.04058116059111416) q[10];
rz(-1.44897315708867) q[10];
ry(-0.027077623450374344) q[11];
rz(-1.7839164696648202) q[11];
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
ry(1.5716414561074277) q[0];
rz(-2.610010645455311) q[0];
ry(-1.5728679034913178) q[1];
rz(2.861016578301916) q[1];
ry(1.5707732270608545) q[2];
rz(-2.155169970925814) q[2];
ry(1.5715163700130657) q[3];
rz(3.014273033138263) q[3];
ry(-1.5732844735437814) q[4];
rz(2.5197442202067792) q[4];
ry(1.5804376194591656) q[5];
rz(1.4428245169094458) q[5];
ry(-1.5703530746273708) q[6];
rz(0.9389906468734422) q[6];
ry(1.6487837792326585) q[7];
rz(-2.0412848330872393) q[7];
ry(1.5791930352845622) q[8];
rz(2.656010752076288) q[8];
ry(1.5319597090001822) q[9];
rz(-2.6259601835771598) q[9];
ry(-1.570138573691195) q[10];
rz(-2.2087850491628087) q[10];
ry(0.7192455860614281) q[11];
rz(-1.13354912718165) q[11];