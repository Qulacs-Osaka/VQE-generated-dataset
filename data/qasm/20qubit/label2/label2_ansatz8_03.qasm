OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.6327726161839111) q[0];
ry(0.9601074328689878) q[1];
cx q[0],q[1];
ry(1.5132758277340868) q[0];
ry(-2.677293983655311) q[1];
cx q[0],q[1];
ry(1.8968531569111722) q[2];
ry(2.212910949804505) q[3];
cx q[2],q[3];
ry(0.500938674320947) q[2];
ry(-0.3893485458220489) q[3];
cx q[2],q[3];
ry(3.0024966511382134) q[4];
ry(-0.14957312672769252) q[5];
cx q[4],q[5];
ry(0.48789295408345545) q[4];
ry(1.1789596252226318) q[5];
cx q[4],q[5];
ry(-0.6551393691479349) q[6];
ry(-0.676986654864578) q[7];
cx q[6],q[7];
ry(2.892500471921786) q[6];
ry(2.5544972130339483) q[7];
cx q[6],q[7];
ry(-0.7777983674269517) q[8];
ry(2.095506050201616) q[9];
cx q[8],q[9];
ry(2.1540814793404808) q[8];
ry(2.707012733882614) q[9];
cx q[8],q[9];
ry(0.06004716035282736) q[10];
ry(1.8873990312993527) q[11];
cx q[10],q[11];
ry(-0.19738875198029862) q[10];
ry(2.6647142514198663) q[11];
cx q[10],q[11];
ry(-0.37914810309946034) q[12];
ry(0.18263735652258759) q[13];
cx q[12],q[13];
ry(-2.303121853974938) q[12];
ry(1.7399717321545571) q[13];
cx q[12],q[13];
ry(0.15557268513159572) q[14];
ry(1.3675745889082176) q[15];
cx q[14],q[15];
ry(-0.3794263259049453) q[14];
ry(-2.949501416278973) q[15];
cx q[14],q[15];
ry(1.558075944042368) q[16];
ry(-1.755940482071183) q[17];
cx q[16],q[17];
ry(-2.917301974479954) q[16];
ry(2.6570458589693535) q[17];
cx q[16],q[17];
ry(-0.9159388157329377) q[18];
ry(1.0249949344014944) q[19];
cx q[18],q[19];
ry(1.92418539307547) q[18];
ry(0.24253641956708893) q[19];
cx q[18],q[19];
ry(-1.3914865800911322) q[0];
ry(1.9893519605089385) q[2];
cx q[0],q[2];
ry(3.0590562888231068) q[0];
ry(-0.032743337135476364) q[2];
cx q[0],q[2];
ry(-2.521533828427731) q[2];
ry(2.0159181529308023) q[4];
cx q[2],q[4];
ry(3.1247056310223167) q[2];
ry(-3.138831842524569) q[4];
cx q[2],q[4];
ry(1.4886907957427213) q[4];
ry(-1.0227454102524531) q[6];
cx q[4],q[6];
ry(3.1410066014744955) q[4];
ry(2.1123196786756018) q[6];
cx q[4],q[6];
ry(-1.2795328211767523) q[6];
ry(-0.5002108497665538) q[8];
cx q[6],q[8];
ry(-1.2164142880600934) q[6];
ry(0.6447819165627969) q[8];
cx q[6],q[8];
ry(0.3174348277062567) q[8];
ry(-0.1454015321512525) q[10];
cx q[8],q[10];
ry(-0.11824714668225646) q[8];
ry(-3.021905107068805) q[10];
cx q[8],q[10];
ry(-2.2592367252265975) q[10];
ry(-2.369297330299182) q[12];
cx q[10],q[12];
ry(0.25301995974731817) q[10];
ry(-0.012711957595326062) q[12];
cx q[10],q[12];
ry(2.693494508381624) q[12];
ry(-0.2689391027828666) q[14];
cx q[12],q[14];
ry(3.0014303179587913) q[12];
ry(0.15088370914458243) q[14];
cx q[12],q[14];
ry(-0.9040000331646556) q[14];
ry(0.24423777645932418) q[16];
cx q[14],q[16];
ry(-0.5225818643778428) q[14];
ry(0.40447742113899904) q[16];
cx q[14],q[16];
ry(0.7110575569715003) q[16];
ry(-1.8980083663326823) q[18];
cx q[16],q[18];
ry(0.003541503288834188) q[16];
ry(0.005129752244769027) q[18];
cx q[16],q[18];
ry(-1.431427772033227) q[1];
ry(-1.2012120348731332) q[3];
cx q[1],q[3];
ry(3.0376864921888402) q[1];
ry(3.1018339372345896) q[3];
cx q[1],q[3];
ry(-2.3970799238956313) q[3];
ry(2.609911680691686) q[5];
cx q[3],q[5];
ry(0.5008541921641845) q[3];
ry(1.1996129145788512) q[5];
cx q[3],q[5];
ry(1.702036956860901) q[5];
ry(-1.1437438038203576) q[7];
cx q[5],q[7];
ry(-0.0003725374415326499) q[5];
ry(-0.00103104579795799) q[7];
cx q[5],q[7];
ry(1.1288790244839575) q[7];
ry(0.3608068217642888) q[9];
cx q[7],q[9];
ry(0.15498320634724114) q[7];
ry(-0.00035740765098957183) q[9];
cx q[7],q[9];
ry(0.11687550694175143) q[9];
ry(-1.5686593974490322) q[11];
cx q[9],q[11];
ry(-1.1362599603520138) q[9];
ry(2.0233895716621397) q[11];
cx q[9],q[11];
ry(1.2229790924839417) q[11];
ry(-1.614436565338739) q[13];
cx q[11],q[13];
ry(-0.8760476671633145) q[11];
ry(-0.7349638537084708) q[13];
cx q[11],q[13];
ry(0.15291192071408236) q[13];
ry(0.9310106182487868) q[15];
cx q[13],q[15];
ry(3.1216428680250563) q[13];
ry(-0.011222449487497066) q[15];
cx q[13],q[15];
ry(-2.1959419838875336) q[15];
ry(2.6009769156636886) q[17];
cx q[15],q[17];
ry(0.03154854747547009) q[15];
ry(-3.1337831093330197) q[17];
cx q[15],q[17];
ry(0.703574235587002) q[17];
ry(-2.7058592277880935) q[19];
cx q[17],q[19];
ry(-0.01130790583407378) q[17];
ry(-0.013366661894645304) q[19];
cx q[17],q[19];
ry(0.7727812834875447) q[0];
ry(-0.8973994398358194) q[1];
cx q[0],q[1];
ry(-0.8443785242726334) q[0];
ry(2.1473858281037854) q[1];
cx q[0],q[1];
ry(0.6048782044458457) q[2];
ry(1.7274539833482665) q[3];
cx q[2],q[3];
ry(-2.032087354983573) q[2];
ry(0.5912924442240843) q[3];
cx q[2],q[3];
ry(-1.7851952035433145) q[4];
ry(-1.873242381931905) q[5];
cx q[4],q[5];
ry(-1.8961995970504013) q[4];
ry(1.8683162908598014) q[5];
cx q[4],q[5];
ry(0.7701917682748967) q[6];
ry(-0.6071964956382958) q[7];
cx q[6],q[7];
ry(-2.9182489443473183) q[6];
ry(0.3420067604837504) q[7];
cx q[6],q[7];
ry(-0.5907387681365834) q[8];
ry(-1.04582045797674) q[9];
cx q[8],q[9];
ry(1.4998867228704613) q[8];
ry(-1.990274106087151) q[9];
cx q[8],q[9];
ry(-0.33088131936930765) q[10];
ry(1.585290287601203) q[11];
cx q[10],q[11];
ry(-2.5258983874643874) q[10];
ry(-0.5923559936123257) q[11];
cx q[10],q[11];
ry(-0.8792193067632927) q[12];
ry(-1.0242226297724177) q[13];
cx q[12],q[13];
ry(1.1339895913915727) q[12];
ry(2.431773573765552) q[13];
cx q[12],q[13];
ry(2.614046825547125) q[14];
ry(1.4873864042485494) q[15];
cx q[14],q[15];
ry(0.04262285125453691) q[14];
ry(0.006674216138344846) q[15];
cx q[14],q[15];
ry(-1.102563472169057) q[16];
ry(2.527296592149437) q[17];
cx q[16],q[17];
ry(1.5212771193532646) q[16];
ry(2.7100854029662766) q[17];
cx q[16],q[17];
ry(0.587231190004473) q[18];
ry(-0.4047599035403923) q[19];
cx q[18],q[19];
ry(-0.7040448256417121) q[18];
ry(2.8815380472128154) q[19];
cx q[18],q[19];
ry(0.2061589472817795) q[0];
ry(-1.4380655051288684) q[2];
cx q[0],q[2];
ry(0.09965692608664391) q[0];
ry(-2.3195449168748534) q[2];
cx q[0],q[2];
ry(-0.3191614532538667) q[2];
ry(-1.3715168437084166) q[4];
cx q[2],q[4];
ry(3.132785212555794) q[2];
ry(2.9930555637379883) q[4];
cx q[2],q[4];
ry(1.9639221653922072) q[4];
ry(2.8529879376904432) q[6];
cx q[4],q[6];
ry(0.001194257529540587) q[4];
ry(-3.1402795677881716) q[6];
cx q[4],q[6];
ry(-0.30815414430968074) q[6];
ry(2.376492389829407) q[8];
cx q[6],q[8];
ry(2.3125913564362035) q[6];
ry(-1.970844425579412) q[8];
cx q[6],q[8];
ry(-1.2416288850381019) q[8];
ry(-1.0290410244834516) q[10];
cx q[8],q[10];
ry(-3.113856664093326) q[8];
ry(0.03393055366255293) q[10];
cx q[8],q[10];
ry(-2.9502866013501046) q[10];
ry(2.6153287436885413) q[12];
cx q[10],q[12];
ry(-3.131228396533638) q[10];
ry(-3.1054100050727236) q[12];
cx q[10],q[12];
ry(-1.9516061697708667) q[12];
ry(-3.0636863005270474) q[14];
cx q[12],q[14];
ry(0.0015585051930440128) q[12];
ry(3.1353844912570974) q[14];
cx q[12],q[14];
ry(3.100235438532064) q[14];
ry(0.0936354225392897) q[16];
cx q[14],q[16];
ry(-2.1730973350411897) q[14];
ry(1.0903558170267136) q[16];
cx q[14],q[16];
ry(0.7111565392952474) q[16];
ry(1.7712806623541788) q[18];
cx q[16],q[18];
ry(2.9053940852022704) q[16];
ry(-3.111326505614741) q[18];
cx q[16],q[18];
ry(0.9922003555281351) q[1];
ry(-1.6990660133752966) q[3];
cx q[1],q[3];
ry(1.3302809008183605) q[1];
ry(0.9240657423701434) q[3];
cx q[1],q[3];
ry(-1.575743020683039) q[3];
ry(-2.264256533478592) q[5];
cx q[3],q[5];
ry(-2.892021896842656) q[3];
ry(-0.36207394457356384) q[5];
cx q[3],q[5];
ry(0.11229055571770065) q[5];
ry(-0.8077434194732178) q[7];
cx q[5],q[7];
ry(-3.1415699120453158) q[5];
ry(0.0003578405385278316) q[7];
cx q[5],q[7];
ry(1.657512069424678) q[7];
ry(-1.6511661351279223) q[9];
cx q[7],q[9];
ry(0.1475991881210339) q[7];
ry(1.3188806255865773) q[9];
cx q[7],q[9];
ry(-0.4720727841742516) q[9];
ry(-0.5263317782455221) q[11];
cx q[9],q[11];
ry(-0.008280427873786508) q[9];
ry(3.0663528371664768) q[11];
cx q[9],q[11];
ry(-2.3435187853363972) q[11];
ry(-1.9803906618678384) q[13];
cx q[11],q[13];
ry(0.11815443276117586) q[11];
ry(-2.7529236734629734) q[13];
cx q[11],q[13];
ry(-2.8601275330628804) q[13];
ry(1.4732218826899617) q[15];
cx q[13],q[15];
ry(0.3720856501906927) q[13];
ry(-0.006113119931733023) q[15];
cx q[13],q[15];
ry(-0.7768814121413552) q[15];
ry(1.8267768463864975) q[17];
cx q[15],q[17];
ry(-3.0905857304084066) q[15];
ry(3.058948772437381) q[17];
cx q[15],q[17];
ry(-2.138305784232479) q[17];
ry(2.9748674974672826) q[19];
cx q[17],q[19];
ry(2.8493026940385464) q[17];
ry(3.140692968679498) q[19];
cx q[17],q[19];
ry(-2.9560655761711496) q[0];
ry(-0.6723754991756491) q[1];
cx q[0],q[1];
ry(0.7447714575920416) q[0];
ry(-2.804543209018615) q[1];
cx q[0],q[1];
ry(-1.8962887518661153) q[2];
ry(0.9761247678965841) q[3];
cx q[2],q[3];
ry(0.7689766763467709) q[2];
ry(3.1178401044714934) q[3];
cx q[2],q[3];
ry(0.4100733053896006) q[4];
ry(-1.358470657917045) q[5];
cx q[4],q[5];
ry(-1.6903605972317237) q[4];
ry(-0.24018731411134464) q[5];
cx q[4],q[5];
ry(0.7004950080659249) q[6];
ry(1.9728547257222913) q[7];
cx q[6],q[7];
ry(2.308938359561969) q[6];
ry(0.23945227357849053) q[7];
cx q[6],q[7];
ry(2.4505203563470905) q[8];
ry(2.323290380994986) q[9];
cx q[8],q[9];
ry(2.477747248492914) q[8];
ry(1.5676266443133824) q[9];
cx q[8],q[9];
ry(0.6254037949483086) q[10];
ry(-3.047307121533696) q[11];
cx q[10],q[11];
ry(-1.7241369627952388) q[10];
ry(0.7507797521109314) q[11];
cx q[10],q[11];
ry(-1.4702565126560394) q[12];
ry(-0.843006804652406) q[13];
cx q[12],q[13];
ry(-0.475750238704423) q[12];
ry(-1.476487108155899) q[13];
cx q[12],q[13];
ry(-2.314071753231819) q[14];
ry(2.2985006645572645) q[15];
cx q[14],q[15];
ry(2.7628132613409693) q[14];
ry(-2.5082604222924743) q[15];
cx q[14],q[15];
ry(-0.06433843650605997) q[16];
ry(0.0752438013014984) q[17];
cx q[16],q[17];
ry(-0.9342446350268646) q[16];
ry(0.49690013339585715) q[17];
cx q[16],q[17];
ry(1.7698456093969597) q[18];
ry(-0.0524694334773269) q[19];
cx q[18],q[19];
ry(2.8383439075235235) q[18];
ry(-0.7562555720262394) q[19];
cx q[18],q[19];
ry(2.6727540379867265) q[0];
ry(-0.2684885722727612) q[2];
cx q[0],q[2];
ry(-0.279867086296715) q[0];
ry(1.8042305410755608) q[2];
cx q[0],q[2];
ry(-0.12924840049258612) q[2];
ry(1.0328305585196071) q[4];
cx q[2],q[4];
ry(2.079384262325) q[2];
ry(1.3744743767568603) q[4];
cx q[2],q[4];
ry(-1.5224103408965535) q[4];
ry(2.7952972008136063) q[6];
cx q[4],q[6];
ry(0.006944356844913458) q[4];
ry(-0.002109939865479582) q[6];
cx q[4],q[6];
ry(1.1218994545041348) q[6];
ry(-1.697186516734632) q[8];
cx q[6],q[8];
ry(3.1200980053170566) q[6];
ry(0.23699888205837463) q[8];
cx q[6],q[8];
ry(2.340429253753588) q[8];
ry(2.195744961878648) q[10];
cx q[8],q[10];
ry(-0.0021192068941360276) q[8];
ry(-3.1362061301288526) q[10];
cx q[8],q[10];
ry(0.43814950133558034) q[10];
ry(0.9517609543428378) q[12];
cx q[10],q[12];
ry(2.5554275341534134) q[10];
ry(1.7368639937002324) q[12];
cx q[10],q[12];
ry(-0.6488757276050521) q[12];
ry(-2.48934982471812) q[14];
cx q[12],q[14];
ry(0.007685235728069093) q[12];
ry(0.0007358844890587498) q[14];
cx q[12],q[14];
ry(-0.46847814394550724) q[14];
ry(1.1323074861411233) q[16];
cx q[14],q[16];
ry(-3.1029666972207304) q[14];
ry(3.130889348976) q[16];
cx q[14],q[16];
ry(2.4377150929533205) q[16];
ry(-1.489636064476306) q[18];
cx q[16],q[18];
ry(0.28512633376871127) q[16];
ry(0.05579540933394162) q[18];
cx q[16],q[18];
ry(-2.0939564970454834) q[1];
ry(1.9360308369925932) q[3];
cx q[1],q[3];
ry(-1.1714694681874587) q[1];
ry(-1.7444846686002498) q[3];
cx q[1],q[3];
ry(-0.22780598931465157) q[3];
ry(0.2927540258452321) q[5];
cx q[3],q[5];
ry(1.4355455243671607) q[3];
ry(0.5038257398339674) q[5];
cx q[3],q[5];
ry(-0.3905637946290532) q[5];
ry(-2.6783951924731855) q[7];
cx q[5],q[7];
ry(-0.004284778377302345) q[5];
ry(-0.004852450211208392) q[7];
cx q[5],q[7];
ry(-0.5367964760686291) q[7];
ry(2.0213342387631057) q[9];
cx q[7],q[9];
ry(-1.2094554221248268) q[7];
ry(-0.7788615956303844) q[9];
cx q[7],q[9];
ry(-3.1141138117333846) q[9];
ry(-1.566806346894313) q[11];
cx q[9],q[11];
ry(-0.0008523676836659837) q[9];
ry(0.006571532087328969) q[11];
cx q[9],q[11];
ry(-2.9928001802110873) q[11];
ry(2.9837782594678712) q[13];
cx q[11],q[13];
ry(2.7327609602081315) q[11];
ry(2.310611496254528) q[13];
cx q[11],q[13];
ry(-1.99548343770591) q[13];
ry(2.1522893932244385) q[15];
cx q[13],q[15];
ry(3.1108815273073427) q[13];
ry(3.141574626447837) q[15];
cx q[13],q[15];
ry(-2.490742365522801) q[15];
ry(-0.017497670293992407) q[17];
cx q[15],q[17];
ry(0.08852215829978181) q[15];
ry(-0.02639788298419216) q[17];
cx q[15],q[17];
ry(2.7418758213798706) q[17];
ry(-0.7381678841560211) q[19];
cx q[17],q[19];
ry(2.8194400039831757) q[17];
ry(-0.05902252977112177) q[19];
cx q[17],q[19];
ry(-2.6933638288191335) q[0];
ry(-1.5513589312714895) q[1];
cx q[0],q[1];
ry(0.5954602775818119) q[0];
ry(0.7107553666197939) q[1];
cx q[0],q[1];
ry(1.8756812784563488) q[2];
ry(-0.7130019448679406) q[3];
cx q[2],q[3];
ry(-2.619567978506865) q[2];
ry(3.0845953321514408) q[3];
cx q[2],q[3];
ry(-2.710344659550005) q[4];
ry(0.07943250983857005) q[5];
cx q[4],q[5];
ry(-1.3423904563004454) q[4];
ry(-2.221598313080336) q[5];
cx q[4],q[5];
ry(1.592311685985582) q[6];
ry(1.9482200585935976) q[7];
cx q[6],q[7];
ry(0.1230621581748439) q[6];
ry(-1.5715428990643259) q[7];
cx q[6],q[7];
ry(1.4719869706197484) q[8];
ry(0.8790086355855153) q[9];
cx q[8],q[9];
ry(-0.9092539052379219) q[8];
ry(-0.888588739012717) q[9];
cx q[8],q[9];
ry(-3.009552864480411) q[10];
ry(-1.9301330832954005) q[11];
cx q[10],q[11];
ry(1.7849547845492095) q[10];
ry(0.01718594847760692) q[11];
cx q[10],q[11];
ry(-0.9529327418727157) q[12];
ry(-1.8142077783554884) q[13];
cx q[12],q[13];
ry(-2.7514892828058084) q[12];
ry(1.5793130220520837) q[13];
cx q[12],q[13];
ry(-1.3818918728887146) q[14];
ry(-0.8061630736844574) q[15];
cx q[14],q[15];
ry(-0.7710114114204067) q[14];
ry(0.056188991576935736) q[15];
cx q[14],q[15];
ry(0.9441289900094488) q[16];
ry(2.778255053246146) q[17];
cx q[16],q[17];
ry(-2.923441464235588) q[16];
ry(2.5955149055005187) q[17];
cx q[16],q[17];
ry(1.6261600583605371) q[18];
ry(-0.3408935486938862) q[19];
cx q[18],q[19];
ry(2.9370341214419193) q[18];
ry(-2.8106980499661924) q[19];
cx q[18],q[19];
ry(-1.58993253842762) q[0];
ry(2.352446676522062) q[2];
cx q[0],q[2];
ry(1.4186963169951443) q[0];
ry(0.6517881342497684) q[2];
cx q[0],q[2];
ry(-0.6210591524781481) q[2];
ry(1.077596538854701) q[4];
cx q[2],q[4];
ry(-3.102411232464785) q[2];
ry(0.013794642536089976) q[4];
cx q[2],q[4];
ry(-0.24214902482401315) q[4];
ry(-1.0964660585303814) q[6];
cx q[4],q[6];
ry(-0.00022224734005418914) q[4];
ry(2.8500379774089346) q[6];
cx q[4],q[6];
ry(0.15281052614918078) q[6];
ry(-1.029712006279043) q[8];
cx q[6],q[8];
ry(1.4972682933732686) q[6];
ry(-0.001148856255909951) q[8];
cx q[6],q[8];
ry(-2.546448867098548) q[8];
ry(-2.636213823514056) q[10];
cx q[8],q[10];
ry(0.0008209128230306661) q[8];
ry(0.004413919530813172) q[10];
cx q[8],q[10];
ry(1.8439914487754128) q[10];
ry(-2.8123167387002925) q[12];
cx q[10],q[12];
ry(-1.8794406018413436) q[10];
ry(-3.062700021531913) q[12];
cx q[10],q[12];
ry(1.5741012097112737) q[12];
ry(1.9008463463375431) q[14];
cx q[12],q[14];
ry(-3.140677962858491) q[12];
ry(-0.001978989391294762) q[14];
cx q[12],q[14];
ry(1.246909353803944) q[14];
ry(-2.5294943738490385) q[16];
cx q[14],q[16];
ry(-0.01831028385588723) q[14];
ry(3.0251097055539877) q[16];
cx q[14],q[16];
ry(-2.56938562912663) q[16];
ry(1.4780848537320401) q[18];
cx q[16],q[18];
ry(-2.7056209444923325) q[16];
ry(-3.0833422658207406) q[18];
cx q[16],q[18];
ry(-0.7768680040604439) q[1];
ry(-0.7944997224414146) q[3];
cx q[1],q[3];
ry(-0.4909835165434373) q[1];
ry(-3.1124866721974436) q[3];
cx q[1],q[3];
ry(-1.3209002858657586) q[3];
ry(2.320785402226667) q[5];
cx q[3],q[5];
ry(3.1074992964549355) q[3];
ry(-0.034796571493557416) q[5];
cx q[3],q[5];
ry(-1.042097193619628) q[5];
ry(-2.175652344018404) q[7];
cx q[5],q[7];
ry(-3.1394038217717015) q[5];
ry(-0.04801761539895999) q[7];
cx q[5],q[7];
ry(-1.9723520711219173) q[7];
ry(2.6188681709829544) q[9];
cx q[7],q[9];
ry(-1.1095989810679745) q[7];
ry(-0.04322216135394407) q[9];
cx q[7],q[9];
ry(0.11022497079600767) q[9];
ry(-2.9063679059072984) q[11];
cx q[9],q[11];
ry(3.137683783377672) q[9];
ry(-0.002783381207454916) q[11];
cx q[9],q[11];
ry(0.34897201934045796) q[11];
ry(2.9907726267757706) q[13];
cx q[11],q[13];
ry(-2.8195577029460583) q[11];
ry(-1.5869392655146983) q[13];
cx q[11],q[13];
ry(-1.8671780957095232) q[13];
ry(0.7120648411461906) q[15];
cx q[13],q[15];
ry(3.1263229465250224) q[13];
ry(-0.010671785736770861) q[15];
cx q[13],q[15];
ry(0.29214973403462263) q[15];
ry(1.7068182746568983) q[17];
cx q[15],q[17];
ry(3.113480106275505) q[15];
ry(0.002286185362547321) q[17];
cx q[15],q[17];
ry(0.3342315045576187) q[17];
ry(-1.3388650395164183) q[19];
cx q[17],q[19];
ry(0.1333522251665311) q[17];
ry(0.05572080153854486) q[19];
cx q[17],q[19];
ry(2.660437020233195) q[0];
ry(-2.2444305333059873) q[1];
cx q[0],q[1];
ry(0.8324544072635435) q[0];
ry(1.9885344249324528) q[1];
cx q[0],q[1];
ry(0.5336981582794535) q[2];
ry(-0.48185580628341285) q[3];
cx q[2],q[3];
ry(0.05880239012917226) q[2];
ry(0.03735238471043978) q[3];
cx q[2],q[3];
ry(3.0261985522525876) q[4];
ry(-0.7221777782431644) q[5];
cx q[4],q[5];
ry(0.0646939333634835) q[4];
ry(-2.773571205649369) q[5];
cx q[4],q[5];
ry(2.4433949670822885) q[6];
ry(2.1912064274095497) q[7];
cx q[6],q[7];
ry(0.11068715637323821) q[6];
ry(3.043516239299303) q[7];
cx q[6],q[7];
ry(0.3277793058976597) q[8];
ry(-2.179405407489873) q[9];
cx q[8],q[9];
ry(-2.8714069059937732) q[8];
ry(-2.6096758328427123) q[9];
cx q[8],q[9];
ry(1.0829640203837325) q[10];
ry(2.0412269539994994) q[11];
cx q[10],q[11];
ry(-1.1738921550350845) q[10];
ry(-0.3398083070850735) q[11];
cx q[10],q[11];
ry(-0.47758641406927443) q[12];
ry(-1.4378310605375786) q[13];
cx q[12],q[13];
ry(-1.097307969598881) q[12];
ry(1.8071375019046139) q[13];
cx q[12],q[13];
ry(2.7402238105419556) q[14];
ry(1.3448832299847924) q[15];
cx q[14],q[15];
ry(0.24108648106155517) q[14];
ry(-0.8308376062806583) q[15];
cx q[14],q[15];
ry(0.008815025598989834) q[16];
ry(-2.543616607391083) q[17];
cx q[16],q[17];
ry(-2.9968283655979087) q[16];
ry(-0.3536016786693029) q[17];
cx q[16],q[17];
ry(1.279343095086758) q[18];
ry(0.11891765039873667) q[19];
cx q[18],q[19];
ry(-1.6940523567780426) q[18];
ry(1.8699222203850598) q[19];
cx q[18],q[19];
ry(-3.0157690711663894) q[0];
ry(-1.5224814982253652) q[2];
cx q[0],q[2];
ry(3.103110245530328) q[0];
ry(0.8413435348995382) q[2];
cx q[0],q[2];
ry(0.28091536959463426) q[2];
ry(-0.4774978453544038) q[4];
cx q[2],q[4];
ry(0.0008980165710789583) q[2];
ry(-0.00017317388443072832) q[4];
cx q[2],q[4];
ry(0.4185469390981522) q[4];
ry(-0.48772175908051985) q[6];
cx q[4],q[6];
ry(2.8472444789005333) q[4];
ry(0.22096043144215116) q[6];
cx q[4],q[6];
ry(0.05309947997809856) q[6];
ry(1.0061598804590608) q[8];
cx q[6],q[8];
ry(-3.1352357780913263) q[6];
ry(-3.1343006778135742) q[8];
cx q[6],q[8];
ry(1.6149121133100337) q[8];
ry(-1.783634737716297) q[10];
cx q[8],q[10];
ry(0.024323809257202986) q[8];
ry(3.138626885895794) q[10];
cx q[8],q[10];
ry(-1.741286981364745) q[10];
ry(-1.23369660422754) q[12];
cx q[10],q[12];
ry(0.2580121706854097) q[10];
ry(-1.7474928212300025) q[12];
cx q[10],q[12];
ry(-2.415391544893407) q[12];
ry(1.3328280855915535) q[14];
cx q[12],q[14];
ry(3.1305488042840497) q[12];
ry(-0.08279769136054949) q[14];
cx q[12],q[14];
ry(-2.188366177739021) q[14];
ry(-1.5325810630806487) q[16];
cx q[14],q[16];
ry(-0.053959628258667215) q[14];
ry(-0.02472834171338195) q[16];
cx q[14],q[16];
ry(2.2863292766558767) q[16];
ry(-1.4791444698463652) q[18];
cx q[16],q[18];
ry(3.071800149557414) q[16];
ry(-0.0059019194706541175) q[18];
cx q[16],q[18];
ry(0.6608114552683737) q[1];
ry(0.20549846774966163) q[3];
cx q[1],q[3];
ry(-2.113184642537606) q[1];
ry(-2.524589877777401) q[3];
cx q[1],q[3];
ry(1.6512556481280285) q[3];
ry(-0.5098302139050892) q[5];
cx q[3],q[5];
ry(0.05230052624779656) q[3];
ry(-3.1379960309412205) q[5];
cx q[3],q[5];
ry(0.4202009679985315) q[5];
ry(1.315945934875222) q[7];
cx q[5],q[7];
ry(-0.09568890055388038) q[5];
ry(2.7660824084285127) q[7];
cx q[5],q[7];
ry(1.138132492182229) q[7];
ry(1.483500403978603) q[9];
cx q[7],q[9];
ry(0.018949505008753142) q[7];
ry(-3.1391767958035173) q[9];
cx q[7],q[9];
ry(1.8829562971439724) q[9];
ry(-0.7158844441011842) q[11];
cx q[9],q[11];
ry(0.004486489265783828) q[9];
ry(-2.892688397109808) q[11];
cx q[9],q[11];
ry(-0.29122834437169764) q[11];
ry(-1.5513860442703256) q[13];
cx q[11],q[13];
ry(1.6984165891051275) q[11];
ry(-0.013162779656965993) q[13];
cx q[11],q[13];
ry(-0.4643344058216403) q[13];
ry(1.1236922317466385) q[15];
cx q[13],q[15];
ry(-3.140239567990144) q[13];
ry(-0.05643254626798923) q[15];
cx q[13],q[15];
ry(1.031148739206032) q[15];
ry(-0.3321352273733673) q[17];
cx q[15],q[17];
ry(3.1222284189585223) q[15];
ry(3.1290707025658335) q[17];
cx q[15],q[17];
ry(2.8616467762330142) q[17];
ry(1.0518909238962166) q[19];
cx q[17],q[19];
ry(-1.5935693594702904) q[17];
ry(-3.1226996335823753) q[19];
cx q[17],q[19];
ry(-2.494606425282516) q[0];
ry(1.8416828391516047) q[1];
cx q[0],q[1];
ry(-1.3684733223773318) q[0];
ry(1.0591934935508602) q[1];
cx q[0],q[1];
ry(-0.5217203641464555) q[2];
ry(-2.822915876853624) q[3];
cx q[2],q[3];
ry(-1.5598598219981472) q[2];
ry(3.0652320155999337) q[3];
cx q[2],q[3];
ry(2.315658916303664) q[4];
ry(-1.0219437162128893) q[5];
cx q[4],q[5];
ry(-3.0729189891241435) q[4];
ry(0.1038999534252687) q[5];
cx q[4],q[5];
ry(2.747743749949206) q[6];
ry(0.7979149598126254) q[7];
cx q[6],q[7];
ry(3.0224351661044384) q[6];
ry(0.19791443043361312) q[7];
cx q[6],q[7];
ry(-2.60765522230729) q[8];
ry(-0.6750509827351436) q[9];
cx q[8],q[9];
ry(0.15158630922008776) q[8];
ry(-0.22907769961219682) q[9];
cx q[8],q[9];
ry(-1.9419245236579483) q[10];
ry(-3.0095298358426907) q[11];
cx q[10],q[11];
ry(-0.2132851572469188) q[10];
ry(1.4499171768411805) q[11];
cx q[10],q[11];
ry(1.258528122327525) q[12];
ry(1.4797999717411328) q[13];
cx q[12],q[13];
ry(1.6534534831539809) q[12];
ry(0.5327163033191102) q[13];
cx q[12],q[13];
ry(-0.10204486166144998) q[14];
ry(1.5731475376651152) q[15];
cx q[14],q[15];
ry(-1.2207244663048626) q[14];
ry(2.8118147061648524) q[15];
cx q[14],q[15];
ry(2.445389052788727) q[16];
ry(-0.20520589754181096) q[17];
cx q[16],q[17];
ry(0.0017956433125752972) q[16];
ry(2.8571372602262257) q[17];
cx q[16],q[17];
ry(2.186708992718688) q[18];
ry(-1.575392703205441) q[19];
cx q[18],q[19];
ry(-1.2727399207095162) q[18];
ry(0.11649724762605906) q[19];
cx q[18],q[19];
ry(-0.8262239530625957) q[0];
ry(2.2779648602989777) q[2];
cx q[0],q[2];
ry(3.1104040392284946) q[0];
ry(-2.265624822291173) q[2];
cx q[0],q[2];
ry(-0.24416981553243775) q[2];
ry(1.9495828620864626) q[4];
cx q[2],q[4];
ry(0.008183550018173946) q[2];
ry(-0.024046754548481708) q[4];
cx q[2],q[4];
ry(1.7445517132165262) q[4];
ry(-2.674790589588685) q[6];
cx q[4],q[6];
ry(-0.2956352174684129) q[4];
ry(-3.0666622381566078) q[6];
cx q[4],q[6];
ry(-0.6321955706284379) q[6];
ry(0.05029953635614248) q[8];
cx q[6],q[8];
ry(0.009033094247587631) q[6];
ry(-0.004105001460103175) q[8];
cx q[6],q[8];
ry(-0.5724388567045128) q[8];
ry(2.1967588254859622) q[10];
cx q[8],q[10];
ry(0.00230498765834497) q[8];
ry(-3.112516187450328) q[10];
cx q[8],q[10];
ry(-0.5860494219257699) q[10];
ry(-1.3867981828388158) q[12];
cx q[10],q[12];
ry(0.5496686904915418) q[10];
ry(-3.1372439855145506) q[12];
cx q[10],q[12];
ry(0.8263275997550394) q[12];
ry(2.3314954815487265) q[14];
cx q[12],q[14];
ry(3.1231633820002127) q[12];
ry(-3.137526796630608) q[14];
cx q[12],q[14];
ry(-0.6621121138245636) q[14];
ry(1.7966767766139622) q[16];
cx q[14],q[16];
ry(3.103482754353329) q[14];
ry(0.02142332848925399) q[16];
cx q[14],q[16];
ry(-1.8474611243023196) q[16];
ry(-2.877765494615043) q[18];
cx q[16],q[18];
ry(0.045056114112171794) q[16];
ry(0.02594202643106775) q[18];
cx q[16],q[18];
ry(-2.0936156711716647) q[1];
ry(-1.9267012484412946) q[3];
cx q[1],q[3];
ry(-2.801537217545873) q[1];
ry(-0.030332388250990935) q[3];
cx q[1],q[3];
ry(2.4203028003307288) q[3];
ry(0.25629120495999036) q[5];
cx q[3],q[5];
ry(3.1312319043659365) q[3];
ry(-3.141546252554734) q[5];
cx q[3],q[5];
ry(1.1938381282149066) q[5];
ry(-0.4293244664150381) q[7];
cx q[5],q[7];
ry(-0.0995848666490579) q[5];
ry(-0.39635557700067636) q[7];
cx q[5],q[7];
ry(0.35298872664591663) q[7];
ry(1.1029104636547846) q[9];
cx q[7],q[9];
ry(-0.009095620169516686) q[7];
ry(3.1371384139581036) q[9];
cx q[7],q[9];
ry(-0.4335442258037366) q[9];
ry(-0.14311082030779865) q[11];
cx q[9],q[11];
ry(-3.128351847050367) q[9];
ry(-2.899688193791066) q[11];
cx q[9],q[11];
ry(2.007442586230236) q[11];
ry(3.056173858635768) q[13];
cx q[11],q[13];
ry(0.03770277477637192) q[11];
ry(-3.141022224860106) q[13];
cx q[11],q[13];
ry(-0.7143584055477694) q[13];
ry(1.7890811891117586) q[15];
cx q[13],q[15];
ry(3.125522971242596) q[13];
ry(-0.007573912014836631) q[15];
cx q[13],q[15];
ry(1.0805554133056836) q[15];
ry(-0.7836838548939785) q[17];
cx q[15],q[17];
ry(-3.13784792354391) q[15];
ry(-3.122572636660816) q[17];
cx q[15],q[17];
ry(2.7273507942726276) q[17];
ry(1.4969611596032044) q[19];
cx q[17],q[19];
ry(-1.7239858113288182) q[17];
ry(3.1064434553324007) q[19];
cx q[17],q[19];
ry(-2.77686209664345) q[0];
ry(-2.2652631806618833) q[1];
ry(2.0934985535279305) q[2];
ry(-1.4604670509086128) q[3];
ry(-0.9032372673782957) q[4];
ry(-2.865567841034331) q[5];
ry(0.7319011871652432) q[6];
ry(1.8281564756934188) q[7];
ry(-0.9019838147776609) q[8];
ry(-2.7588837245156244) q[9];
ry(0.7901894325999104) q[10];
ry(1.4037286321241291) q[11];
ry(-2.123093603242248) q[12];
ry(3.1369022270923512) q[13];
ry(-0.8159952922629508) q[14];
ry(1.846317713980988) q[15];
ry(0.42450177295495745) q[16];
ry(1.048628636248089) q[17];
ry(-1.8491300595076767) q[18];
ry(-2.171627026293132) q[19];