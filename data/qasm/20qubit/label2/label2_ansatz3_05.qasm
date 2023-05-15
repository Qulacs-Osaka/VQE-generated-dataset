OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.018090445285325912) q[0];
rz(0.5548429239801244) q[0];
ry(2.9954466611240473) q[1];
rz(-0.6916607828006766) q[1];
ry(1.545779072610502) q[2];
rz(0.3616872829190781) q[2];
ry(0.1054482939655288) q[3];
rz(-1.4448099155999752) q[3];
ry(-0.10819513968338512) q[4];
rz(1.9526408421579065) q[4];
ry(1.2720351478303842) q[5];
rz(-0.28330583393419234) q[5];
ry(-7.24384729966267e-05) q[6];
rz(0.9320140518888183) q[6];
ry(0.0007807249906494991) q[7];
rz(-1.9626646445181974) q[7];
ry(-2.532420243840457) q[8];
rz(-2.187963633159555) q[8];
ry(3.123648161612512) q[9];
rz(0.5016918414387165) q[9];
ry(0.00044941975734713526) q[10];
rz(-0.9201104120038862) q[10];
ry(0.44383322296415373) q[11];
rz(0.0872338440696203) q[11];
ry(0.33734122791869847) q[12];
rz(0.1298445616883077) q[12];
ry(-3.140303328322352) q[13];
rz(1.563968431767095) q[13];
ry(7.821457339574067e-05) q[14];
rz(0.47949955896356333) q[14];
ry(-0.5298991396027654) q[15];
rz(-0.7626393525257256) q[15];
ry(-1.5948308660246682) q[16];
rz(1.2262462872826534) q[16];
ry(0.9461868164210749) q[17];
rz(3.1305288040504338) q[17];
ry(3.1398428267955496) q[18];
rz(-1.9740553598403991) q[18];
ry(1.5683666045201035) q[19];
rz(-1.568956251106651) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(0.000407197349956904) q[0];
rz(2.4645310916765215) q[0];
ry(3.1000399562616963) q[1];
rz(2.8316447386253527) q[1];
ry(0.07275207027856115) q[2];
rz(2.7759428324553106) q[2];
ry(0.00148865203853088) q[3];
rz(-0.1512545300745976) q[3];
ry(1.5795537673998403) q[4];
rz(-0.1986211761565073) q[4];
ry(-2.8271061890921607) q[5];
rz(-2.2283637535084737) q[5];
ry(0.0006512663198483537) q[6];
rz(-1.4791221027837613) q[6];
ry(3.141476269482708) q[7];
rz(-1.868419234874163) q[7];
ry(1.5453128185644651) q[8];
rz(-0.49192060103570956) q[8];
ry(0.010857630811716312) q[9];
rz(-3.045769677471072) q[9];
ry(0.00026940545115797183) q[10];
rz(-2.263798245862035) q[10];
ry(-2.2720793587136687) q[11];
rz(1.3554583642187223) q[11];
ry(0.03081650223840402) q[12];
rz(0.9597255042005352) q[12];
ry(3.141368068575572) q[13];
rz(-1.8087774072339267) q[13];
ry(3.1415657219330684) q[14];
rz(-2.0695196294379814) q[14];
ry(0.0001471452728987188) q[15];
rz(2.428149840859671) q[15];
ry(2.6254305998717467) q[16];
rz(1.2616633255584135) q[16];
ry(3.141116127832433) q[17];
rz(-0.9600924069141168) q[17];
ry(-1.570826237307615) q[18];
rz(2.4930422857671815) q[18];
ry(2.5709244831067886) q[19];
rz(-1.1004138931916296) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-0.006378411361223079) q[0];
rz(-2.947576474082937) q[0];
ry(1.4522277795100376) q[1];
rz(1.5974939133021049) q[1];
ry(-1.555762094634095) q[2];
rz(-0.6706203910175715) q[2];
ry(0.5366910053349461) q[3];
rz(1.758372348329961) q[3];
ry(-2.2998141184421756) q[4];
rz(-0.6231571897294002) q[4];
ry(-1.956199966126725) q[5];
rz(1.8392752631526956) q[5];
ry(-0.05397915708484178) q[6];
rz(1.0737195752325068) q[6];
ry(0.1782935347970309) q[7];
rz(-0.5800915233729628) q[7];
ry(-2.455032820106114) q[8];
rz(0.11371497783170771) q[8];
ry(-1.1311907498359914) q[9];
rz(1.1498801932133924) q[9];
ry(0.00037368705110552014) q[10];
rz(2.9090890383285153) q[10];
ry(-0.19335939540673458) q[11];
rz(1.0408289198139258) q[11];
ry(-1.7129622564968603) q[12];
rz(-0.22319440695571213) q[12];
ry(-1.6719958506394585) q[13];
rz(0.6030224716716379) q[13];
ry(-0.0001475982425951372) q[14];
rz(1.7009061599019488) q[14];
ry(1.6259726918780055) q[15];
rz(-0.5124865559151823) q[15];
ry(-1.570750455431289) q[16];
rz(-0.6762162145514446) q[16];
ry(-0.6844316734323455) q[17];
rz(-1.4255559118554577) q[17];
ry(3.133684134493109e-05) q[18];
rz(1.440285043245093) q[18];
ry(-3.1385505054240235) q[19];
rz(2.116731437786595) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.5706651724292577) q[0];
rz(-2.0502292723086635) q[0];
ry(-3.1339464679275055) q[1];
rz(1.6394899179841145) q[1];
ry(-0.1127615864879639) q[2];
rz(-2.2604732780367307) q[2];
ry(-0.0012532137541345634) q[3];
rz(-0.09037829634930804) q[3];
ry(3.122482689740432) q[4];
rz(1.1301759362041341) q[4];
ry(-3.140346558498182) q[5];
rz(-2.763496731741415) q[5];
ry(3.1408800110591892) q[6];
rz(0.9666718180710521) q[6];
ry(-1.8017985943252768e-05) q[7];
rz(2.67886354050819) q[7];
ry(2.223203996799515) q[8];
rz(-1.1102134582777812) q[8];
ry(-0.009540277658216745) q[9];
rz(-1.0688716272172156) q[9];
ry(-3.133533599772067) q[10];
rz(-1.4084679821729509) q[10];
ry(0.0018087826631434081) q[11];
rz(0.8303546889390913) q[11];
ry(-1.298193118881225) q[12];
rz(0.44233451048700534) q[12];
ry(3.14153790111964) q[13];
rz(0.474249542599642) q[13];
ry(-1.5706248759090133) q[14];
rz(-1.836759938209772) q[14];
ry(1.5707032191073624) q[15];
rz(-1.570846510668468) q[15];
ry(0.7549030327368871) q[16];
rz(3.002334481414646) q[16];
ry(3.141449535896405) q[17];
rz(-1.2300022120378502) q[17];
ry(1.5108432011563646) q[18];
rz(0.7326638797701017) q[18];
ry(1.3566335573318582) q[19];
rz(-1.4409806043717506) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(1.653922508941144) q[0];
rz(0.193954175884509) q[0];
ry(1.571153902921382) q[1];
rz(0.5278753823854262) q[1];
ry(3.14141952894733) q[2];
rz(1.7407613763180187) q[2];
ry(1.5626293169107928) q[3];
rz(2.60828415677153) q[3];
ry(-1.2987991957311964) q[4];
rz(2.8593908632813525) q[4];
ry(1.1578233601345962) q[5];
rz(-0.6277892214311295) q[5];
ry(0.08444355850673012) q[6];
rz(-2.78087335154787) q[6];
ry(-2.8556648085751064) q[7];
rz(-2.407766027856994) q[7];
ry(0.8073525250239972) q[8];
rz(2.4151192284643574) q[8];
ry(-0.6555140596740692) q[9];
rz(-0.752326776267057) q[9];
ry(-0.0038647372298445995) q[10];
rz(0.7487229188724571) q[10];
ry(-1.3586725876782901) q[11];
rz(-2.2997068528320863) q[11];
ry(-0.00011305457050880818) q[12];
rz(0.6667680179621666) q[12];
ry(-3.1414925293159928) q[13];
rz(2.357962181905275) q[13];
ry(-0.00021458212487551255) q[14];
rz(1.8713589201825878) q[14];
ry(1.5708338907705934) q[15];
rz(1.6950660228414722) q[15];
ry(0.30448743862738953) q[16];
rz(3.140760592111188) q[16];
ry(0.0001957264014533757) q[17];
rz(-0.29979008273190605) q[17];
ry(-0.0020508199727693518) q[18];
rz(2.7395399218751586) q[18];
ry(-0.10255399034681671) q[19];
rz(0.11035539879022002) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-3.1113280929294485) q[0];
rz(-1.5815438809342797) q[0];
ry(-1.8358859164315309) q[1];
rz(1.8750665577120154) q[1];
ry(-0.000715049211984431) q[2];
rz(-2.7695041423933078) q[2];
ry(-1.5706751490262993) q[3];
rz(-2.568536585918277) q[3];
ry(1.5900220553421445) q[4];
rz(-1.547995082594259) q[4];
ry(-3.136761847265873) q[5];
rz(2.462347437547558) q[5];
ry(0.0198177243890298) q[6];
rz(-0.5163734856036744) q[6];
ry(3.1415579809923457) q[7];
rz(0.45506530451654204) q[7];
ry(-2.5877349625864206) q[8];
rz(-1.5346668576232707) q[8];
ry(2.875867075083217) q[9];
rz(-0.9553917809207393) q[9];
ry(-0.0007401646485831748) q[10];
rz(-1.274241341919761) q[10];
ry(-0.0007743427546767354) q[11];
rz(-0.1617713245229765) q[11];
ry(0.5604598293317757) q[12];
rz(0.872574725076637) q[12];
ry(0.002676269102555473) q[13];
rz(1.3540743560541348) q[13];
ry(3.130787893443976) q[14];
rz(1.7920050098722033) q[14];
ry(0.11884025661095877) q[15];
rz(1.4458167271672429) q[15];
ry(-0.6533291307597455) q[16];
rz(-0.4463457664335424) q[16];
ry(-1.5701187735655742) q[17];
rz(3.13074325854412) q[17];
ry(3.0901550203237544) q[18];
rz(0.4058243031616586) q[18];
ry(0.0788206161809281) q[19];
rz(-0.08864221872070033) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.336025904995612) q[0];
rz(3.1168398630995995) q[0];
ry(0.2942549489096189) q[1];
rz(0.7875339008957765) q[1];
ry(0.025470436067208) q[2];
rz(-1.7225662673297444) q[2];
ry(-3.1162082932654718) q[3];
rz(0.2775781159055164) q[3];
ry(2.108958735519144) q[4];
rz(-2.9946146980482906) q[4];
ry(-1.5713239739963718) q[5];
rz(-1.5272543557871054) q[5];
ry(-3.1411651108427563) q[6];
rz(-1.057264912291526) q[6];
ry(0.294273931764093) q[7];
rz(2.1385177514365967) q[7];
ry(2.757205854337141) q[8];
rz(1.0446252975646466) q[8];
ry(1.4008997957282827) q[9];
rz(-0.4253794405666991) q[9];
ry(0.0899901319295866) q[10];
rz(-2.197945449290542) q[10];
ry(-0.008904989310516977) q[11];
rz(1.8455929528801198) q[11];
ry(-5.082350267607645e-05) q[12];
rz(-3.067699637303573) q[12];
ry(5.595785410263627e-05) q[13];
rz(2.035268353451326) q[13];
ry(1.5710745175334266) q[14];
rz(1.5708699140146545) q[14];
ry(1.521375648186512) q[15];
rz(1.5706219585293273) q[15];
ry(4.965847119375866e-05) q[16];
rz(-2.6740999549298405) q[16];
ry(3.1311820870829554) q[17];
rz(-1.4598783534999535) q[17];
ry(-1.6729615647199) q[18];
rz(-2.55431274284721) q[18];
ry(-1.5299982860820762) q[19];
rz(-1.5447787259959587) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(2.8504952407038733) q[0];
rz(-1.095796480891126) q[0];
ry(-0.795252540963655) q[1];
rz(-1.9149911627759817) q[1];
ry(-3.140996842185881) q[2];
rz(0.9028120969388667) q[2];
ry(-3.141441299560066) q[3];
rz(2.364600925564922) q[3];
ry(-1.5710168950230126) q[4];
rz(-0.24526341368776364) q[4];
ry(0.001353672558875793) q[5];
rz(-2.1480505868759394) q[5];
ry(3.1028648897090565) q[6];
rz(1.1820016709939578) q[6];
ry(0.0008032284550170132) q[7];
rz(-1.3738657198028303) q[7];
ry(1.676604070672684) q[8];
rz(1.3814716167070449) q[8];
ry(2.664608987526654) q[9];
rz(-2.7197821923822856) q[9];
ry(-0.0005298331243697163) q[10];
rz(0.8511508380212831) q[10];
ry(-3.1408375245192484) q[11];
rz(-0.3921924831396186) q[11];
ry(3.0946532633972317) q[12];
rz(-0.27044061811420167) q[12];
ry(0.047603386820712905) q[13];
rz(-1.8491246226338465) q[13];
ry(-1.5707483892482932) q[14];
rz(-1.0206465782416905) q[14];
ry(-1.5707982319860108) q[15];
rz(-0.17092145265138942) q[15];
ry(-1.5702212951730894) q[16];
rz(-2.550916768236039) q[16];
ry(-3.141565123857906) q[17];
rz(0.5685917695441444) q[17];
ry(-1.5241569448900574) q[18];
rz(-0.8558235404392923) q[18];
ry(1.6652934000945532) q[19];
rz(2.7549114600888713) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
ry(-1.9362733488934578) q[0];
rz(-2.5734136426326417) q[0];
ry(2.4924526042099098) q[1];
rz(2.9644382686421804) q[1];
ry(1.8082110093759267) q[2];
rz(3.0775174185891188) q[2];
ry(-1.6119487100650953) q[3];
rz(1.5049000817337208) q[3];
ry(1.9842374169819665) q[4];
rz(1.405775956831915) q[4];
ry(0.4516777808554071) q[5];
rz(1.9945075253176592) q[5];
ry(-1.3801928235561105) q[6];
rz(3.0769757454725015) q[6];
ry(1.6449187297156231) q[7];
rz(3.135154594278927) q[7];
ry(-1.8987155598522616) q[8];
rz(-0.10850441976711699) q[8];
ry(-1.198784542522823) q[9];
rz(-0.038778958667556644) q[9];
ry(-1.3947414082034966) q[10];
rz(3.0548060479522507) q[10];
ry(-1.4537255014298183) q[11];
rz(3.0437821339365585) q[11];
ry(1.427986156425634) q[12];
rz(-0.06766941179488604) q[12];
ry(-1.4278455853542873) q[13];
rz(3.0739876169861295) q[13];
ry(2.822568551549653) q[14];
rz(-1.0875605538986335) q[14];
ry(1.8417696619761648) q[15];
rz(1.4807156707677454) q[15];
ry(-1.292839185634171) q[16];
rz(1.5394711004366215) q[16];
ry(-2.8157453617507415) q[17];
rz(2.10514252658543) q[17];
ry(1.2729562164196835) q[18];
rz(-1.5309383667397658) q[18];
ry(-0.8956197719838825) q[19];
rz(-2.8480579677941456) q[19];