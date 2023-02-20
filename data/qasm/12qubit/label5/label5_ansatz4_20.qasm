OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.6066882751556093) q[0];
rz(-1.3655262573289741) q[0];
ry(-1.4604883293196709) q[1];
rz(-1.9702944279418046) q[1];
ry(-0.3685854890568283) q[2];
rz(-2.3120512097139967) q[2];
ry(1.0418293702494932) q[3];
rz(-1.6895077707509978) q[3];
ry(2.113337343029011) q[4];
rz(-1.2149670301766209) q[4];
ry(0.38952243631283956) q[5];
rz(-3.115522869124697) q[5];
ry(0.0018545111954475885) q[6];
rz(-2.7441454598428825) q[6];
ry(3.1353693135543983) q[7];
rz(0.5395659424973667) q[7];
ry(-0.32814274186990894) q[8];
rz(0.2626734823133834) q[8];
ry(-1.7877315017446485) q[9];
rz(-1.9713138497652904) q[9];
ry(-0.8869348517556387) q[10];
rz(1.9650041209643225) q[10];
ry(1.6713650345148416) q[11];
rz(-2.612406258907476) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.7370371502893536) q[0];
rz(-1.8656312159236401) q[0];
ry(1.6741264013196957) q[1];
rz(1.2141010613650793) q[1];
ry(-1.2679215992135806) q[2];
rz(-2.6361134178027137) q[2];
ry(-1.7213423156388994) q[3];
rz(-0.1310782409126263) q[3];
ry(-1.008972603441531) q[4];
rz(0.08570739178724524) q[4];
ry(1.027224191784544) q[5];
rz(-0.5741936926988487) q[5];
ry(-0.012617266382189416) q[6];
rz(-2.3171361280378067) q[6];
ry(-3.1379571793515812) q[7];
rz(0.5886449070161842) q[7];
ry(1.4933162342064983) q[8];
rz(3.0856447818114296) q[8];
ry(-2.734644671909113) q[9];
rz(2.0412390960253455) q[9];
ry(1.720121937806474) q[10];
rz(-1.1859079995772608) q[10];
ry(1.339468345866023) q[11];
rz(-0.947947352025128) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.2421864381181622) q[0];
rz(1.4251364156906536) q[0];
ry(1.538624058643169) q[1];
rz(-1.1981029320323193) q[1];
ry(1.9173194002907223) q[2];
rz(-2.9600088476191724) q[2];
ry(0.5474733881505646) q[3];
rz(-2.3669992906138475) q[3];
ry(0.8675531990239502) q[4];
rz(-1.9262172551092767) q[4];
ry(-1.2829014689599703) q[5];
rz(0.21810060201656079) q[5];
ry(-3.1394877761890427) q[6];
rz(1.0024634905887442) q[6];
ry(-3.141324498174638) q[7];
rz(-2.585176681281685) q[7];
ry(-0.3163990386378236) q[8];
rz(-0.6832911408025247) q[8];
ry(2.231102886019379) q[9];
rz(0.7089127979443584) q[9];
ry(1.6235423331007959) q[10];
rz(0.979062015084268) q[10];
ry(-0.8943668913456033) q[11];
rz(-2.73203209995753) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.781800655909595) q[0];
rz(2.6577682348049905) q[0];
ry(-2.0293778873176205) q[1];
rz(-2.529777934464827) q[1];
ry(0.3717454846993835) q[2];
rz(-2.4203504537083314) q[2];
ry(-0.8371801939805387) q[3];
rz(2.506388838144463) q[3];
ry(2.0445237282221784) q[4];
rz(0.719264523266185) q[4];
ry(-2.932127808597792) q[5];
rz(1.0895569669550618) q[5];
ry(0.020983359423550802) q[6];
rz(0.622468252669266) q[6];
ry(3.1339389139280027) q[7];
rz(-0.5081584769726044) q[7];
ry(-1.9166163751500425) q[8];
rz(0.8068452354342329) q[8];
ry(1.810354317879174) q[9];
rz(3.030121012180886) q[9];
ry(-0.28733820711286745) q[10];
rz(-2.9237059161414836) q[10];
ry(1.3207863657974956) q[11];
rz(2.1035195613649) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.145538715598253) q[0];
rz(-1.0939405954128931) q[0];
ry(-2.2886695818251837) q[1];
rz(-2.041248394203246) q[1];
ry(1.1957841574725374) q[2];
rz(1.1697964832778902) q[2];
ry(2.0328290760105183) q[3];
rz(-2.2087071891554757) q[3];
ry(0.7608273435441193) q[4];
rz(1.491549974982755) q[4];
ry(-2.7761833077733296) q[5];
rz(2.144064896237283) q[5];
ry(-1.577471008268228) q[6];
rz(1.1257923749674605e-05) q[6];
ry(1.5687659738380109) q[7];
rz(-0.0024144174533979844) q[7];
ry(2.0801771692974826) q[8];
rz(0.602439215756589) q[8];
ry(-0.028020271294931835) q[9];
rz(0.11793464458638249) q[9];
ry(-0.9038965433075447) q[10];
rz(-1.0235829650100683) q[10];
ry(-1.1069925538064287) q[11];
rz(-1.9326592051432339) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.9057729327023427) q[0];
rz(-0.9070975001041858) q[0];
ry(0.6968218365761469) q[1];
rz(2.2313645775545883) q[1];
ry(-1.361749400983955) q[2];
rz(-1.933467609678956) q[2];
ry(1.067348594201035) q[3];
rz(1.9067691441213377) q[3];
ry(1.1612796916124637) q[4];
rz(1.8428644487159689) q[4];
ry(0.9331079858831739) q[5];
rz(2.5576675068345045) q[5];
ry(-1.4694073368892262) q[6];
rz(-1.4038056345667513) q[6];
ry(-1.672864336057792) q[7];
rz(0.7981639810958091) q[7];
ry(1.8968503079479175) q[8];
rz(-0.9528612079637145) q[8];
ry(2.939859285167683) q[9];
rz(-0.5675909430683417) q[9];
ry(-2.9935465344534298) q[10];
rz(1.3707927581840398) q[10];
ry(-1.9661234317721052) q[11];
rz(1.7334343337377627) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.8025537395445355) q[0];
rz(-0.846177057427875) q[0];
ry(2.397718237446109) q[1];
rz(-1.4286980954455202) q[1];
ry(2.6219875867722213) q[2];
rz(2.26649847952973) q[2];
ry(-2.4610444759917725) q[3];
rz(0.5640952291527653) q[3];
ry(2.6662143539211494) q[4];
rz(-0.38296741651810356) q[4];
ry(-1.3723582534953227) q[5];
rz(-2.603196605405319) q[5];
ry(-0.19620038779041238) q[6];
rz(1.984251142147381) q[6];
ry(-0.14107550021898288) q[7];
rz(-1.8195981493739088) q[7];
ry(1.778850164768504) q[8];
rz(-1.9061229917911482) q[8];
ry(-0.526659812988805) q[9];
rz(3.1219821252890947) q[9];
ry(0.942490729003932) q[10];
rz(0.023788830861705357) q[10];
ry(-0.21557024821958837) q[11];
rz(1.2601044861253055) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.4606887735710705) q[0];
rz(-1.8516870850507532) q[0];
ry(-0.5986435870112401) q[1];
rz(-0.01434481999986607) q[1];
ry(-1.4553635006994154) q[2];
rz(-1.5396376839367554) q[2];
ry(0.24079743815534765) q[3];
rz(2.288724801658673) q[3];
ry(-0.23143542992831193) q[4];
rz(2.3653980189086448) q[4];
ry(2.9422826038991254) q[5];
rz(-1.7038004424989124) q[5];
ry(-3.139598177557263) q[6];
rz(-0.5555059945965399) q[6];
ry(-0.001431481005092827) q[7];
rz(1.7128081021294124) q[7];
ry(1.9921870021525292) q[8];
rz(0.7158677535160987) q[8];
ry(2.7404574190142155) q[9];
rz(1.3970753220955654) q[9];
ry(0.7667817330953222) q[10];
rz(-0.8527884389797282) q[10];
ry(1.1398392905920292) q[11];
rz(-0.01782127639521125) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5450637780071728) q[0];
rz(-0.058057217975026276) q[0];
ry(0.4741548524356167) q[1];
rz(-1.3016415119807367) q[1];
ry(0.7740582906695944) q[2];
rz(2.129943020237947) q[2];
ry(-0.33199516107670757) q[3];
rz(-2.0150503825324217) q[3];
ry(1.6427359539853177) q[4];
rz(-1.3316267649919065) q[4];
ry(-0.4561640999981157) q[5];
rz(1.3485918268041583) q[5];
ry(2.070539090648146) q[6];
rz(-0.7943853510395044) q[6];
ry(0.17886621940853126) q[7];
rz(-2.9515083129495125) q[7];
ry(-0.39028559377091965) q[8];
rz(-2.685996574483579) q[8];
ry(-0.002959611299111227) q[9];
rz(-0.46010173160932927) q[9];
ry(-1.6908269453742322) q[10];
rz(1.5544099388038317) q[10];
ry(2.304701349935576) q[11];
rz(1.8098215333303254) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.35010790646219764) q[0];
rz(1.0500285675856373) q[0];
ry(1.3696749966021828) q[1];
rz(0.37280093319620944) q[1];
ry(2.7252170138813057) q[2];
rz(1.305495202248399) q[2];
ry(2.806986667122433) q[3];
rz(0.8416939547894889) q[3];
ry(-0.006161597015657548) q[4];
rz(1.0815136354225225) q[4];
ry(-3.1411142442174547) q[5];
rz(-1.283086654349746) q[5];
ry(-0.002536080954130604) q[6];
rz(-0.2022789402442715) q[6];
ry(-3.137258990227773) q[7];
rz(-2.1823133983674228) q[7];
ry(-0.8208721545066863) q[8];
rz(0.49854747626312523) q[8];
ry(-1.7242756082839943) q[9];
rz(2.5927752142169007) q[9];
ry(-1.7094782295211406) q[10];
rz(2.217895980109997) q[10];
ry(1.9048740917387272) q[11];
rz(-2.594633521358102) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.0372110308271214) q[0];
rz(2.2835018030846204) q[0];
ry(2.067741973198707) q[1];
rz(1.4465567590468065) q[1];
ry(0.03449471228045187) q[2];
rz(2.8266511838553603) q[2];
ry(-0.6154351664447084) q[3];
rz(-2.745964229683712) q[3];
ry(2.108756852669794) q[4];
rz(1.6097207869408239) q[4];
ry(1.766457714104555) q[5];
rz(-0.36109117357989795) q[5];
ry(1.7846588013985605) q[6];
rz(-2.1017725382364354) q[6];
ry(1.026320790568196) q[7];
rz(-2.114687071474033) q[7];
ry(-2.386819525663162) q[8];
rz(1.369067071833132) q[8];
ry(-1.4316071141021993) q[9];
rz(2.586807583653994) q[9];
ry(-0.32137450614021434) q[10];
rz(-0.3005396650122118) q[10];
ry(-0.4751444569696064) q[11];
rz(-2.6837538946287807) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.72994447260248) q[0];
rz(-2.0368578922788787) q[0];
ry(-1.0561983941849338) q[1];
rz(-1.7660999945667428) q[1];
ry(1.4551493314561812) q[2];
rz(0.08907880366148076) q[2];
ry(-1.4979738609821363) q[3];
rz(2.294779712529143) q[3];
ry(-3.1360915437027326) q[4];
rz(2.007112078526007) q[4];
ry(0.005567886159068856) q[5];
rz(2.3562628784209654) q[5];
ry(0.0024296734738626924) q[6];
rz(-1.4641749860321793) q[6];
ry(3.138482952433667) q[7];
rz(0.8422858373944063) q[7];
ry(-0.09767064353533961) q[8];
rz(0.6547392527001679) q[8];
ry(-2.0173883469622056) q[9];
rz(0.49323412821880197) q[9];
ry(-0.9788178951971518) q[10];
rz(-2.8704447284786827) q[10];
ry(-0.5422090588346937) q[11];
rz(-1.8400715103415966) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.1941656209295735) q[0];
rz(2.3316815897809606) q[0];
ry(-0.43581606118861593) q[1];
rz(3.02155092961335) q[1];
ry(-2.3162494854578823) q[2];
rz(-2.647655921556588) q[2];
ry(1.59311589521239) q[3];
rz(-1.8173866629789908) q[3];
ry(0.2534077249913279) q[4];
rz(2.8763348200211647) q[4];
ry(-1.472846226772677) q[5];
rz(3.0364509605299346) q[5];
ry(-1.6114162620541719) q[6];
rz(0.20110680254352553) q[6];
ry(0.9198526411625504) q[7];
rz(-0.2717321374678596) q[7];
ry(-2.141056013319732) q[8];
rz(2.0059440259103694) q[8];
ry(2.014864324908782) q[9];
rz(1.9238923091733864) q[9];
ry(-2.2440709908556276) q[10];
rz(1.0223380927877421) q[10];
ry(-2.5645208891369347) q[11];
rz(-1.5022051162856758) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.3144648505967385) q[0];
rz(0.2814118184176643) q[0];
ry(0.013220807860164108) q[1];
rz(2.749038766396669) q[1];
ry(-1.9455609309522002) q[2];
rz(0.9600795836973132) q[2];
ry(1.3700046653611757) q[3];
rz(0.45523987342301553) q[3];
ry(3.141461730625041) q[4];
rz(1.1852700672970782) q[4];
ry(-3.139691258748498) q[5];
rz(-2.2853308155753784) q[5];
ry(-3.1405841240169066) q[6];
rz(-2.6952216917892526) q[6];
ry(-0.0012570044071837927) q[7];
rz(-2.4912419355117383) q[7];
ry(2.0183302499450324) q[8];
rz(-1.38929771226109) q[8];
ry(-0.7788955646020735) q[9];
rz(-0.8636967747373642) q[9];
ry(-1.1963978728204347) q[10];
rz(0.8739064733628313) q[10];
ry(2.8629985684503305) q[11];
rz(-2.7498540498378845) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.1397334315022745) q[0];
rz(-0.16653715538440955) q[0];
ry(1.9820596064995772) q[1];
rz(1.2347601682021607) q[1];
ry(3.1114729460492443) q[2];
rz(0.9768924720628709) q[2];
ry(2.311976495125672) q[3];
rz(-2.6422111554046808) q[3];
ry(2.3490031189244696) q[4];
rz(-2.1871618717231516) q[4];
ry(1.728871522393672) q[5];
rz(2.2095752965169932) q[5];
ry(-1.4541553246185313) q[6];
rz(-2.781389552911675) q[6];
ry(-1.383157024333749) q[7];
rz(-0.35916830828738033) q[7];
ry(-0.7170021623912161) q[8];
rz(2.933920581177502) q[8];
ry(1.4797586656484707) q[9];
rz(2.8865114266101566) q[9];
ry(-0.03157625243704261) q[10];
rz(-2.5300232296606158) q[10];
ry(1.9864070267511522) q[11];
rz(-0.8832006135994926) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.40412181932393487) q[0];
rz(-1.7590933707570517) q[0];
ry(-2.392605984688343) q[1];
rz(-2.55262473759573) q[1];
ry(-0.590135188000877) q[2];
rz(-1.8386751547120326) q[2];
ry(-2.4834670315137655) q[3];
rz(-0.6558130528421467) q[3];
ry(1.571646307031891) q[4];
rz(1.5870784772147026) q[4];
ry(-1.6639973552499088) q[5];
rz(-1.5713170958731704) q[5];
ry(1.5694171784945734) q[6];
rz(0.12879524557584277) q[6];
ry(-1.5679838160039394) q[7];
rz(-2.5834860542199967) q[7];
ry(0.589655878395601) q[8];
rz(1.1700989928207697) q[8];
ry(-2.186081250337688) q[9];
rz(-1.4213373831001332) q[9];
ry(-1.8520539658149766) q[10];
rz(2.723155914928502) q[10];
ry(-2.8635943395155024) q[11];
rz(1.2532736332252856) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.24653740748914) q[0];
rz(2.160319012039273) q[0];
ry(-1.8126013714168867) q[1];
rz(0.5850708702910197) q[1];
ry(-2.999226739861837) q[2];
rz(2.5206261744593315) q[2];
ry(-2.956153230391096) q[3];
rz(0.5922505798692708) q[3];
ry(1.567707877001621) q[4];
rz(-1.5716378027007958) q[4];
ry(1.5706237255623607) q[5];
rz(-2.469656209638693) q[5];
ry(0.004383436893340864) q[6];
rz(-2.442537937754023) q[6];
ry(-3.140781117757766) q[7];
rz(3.076433224257725) q[7];
ry(1.4902125464179097) q[8];
rz(0.7868665597931841) q[8];
ry(1.6939278433927383) q[9];
rz(-3.1388909421594455) q[9];
ry(-0.6715075047913679) q[10];
rz(-2.467285802954995) q[10];
ry(-2.4640243026715973) q[11];
rz(2.146390172819075) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.444434534497942) q[0];
rz(2.183199903943939) q[0];
ry(0.009971698305816994) q[1];
rz(-2.0624381060203554) q[1];
ry(1.8527687786954898) q[2];
rz(2.8379503951908487) q[2];
ry(-0.41093213876278867) q[3];
rz(-0.3386744471717135) q[3];
ry(1.4721637636026488) q[4];
rz(1.570137707974518) q[4];
ry(0.00029757892121538523) q[5];
rz(2.474313825051446) q[5];
ry(-0.01068130585849758) q[6];
rz(0.11936133178650689) q[6];
ry(0.00827662455431799) q[7];
rz(2.5269147776643375) q[7];
ry(-2.9069480288728613) q[8];
rz(-2.3166638511770232) q[8];
ry(-2.4874757445785134) q[9];
rz(-2.483769006555105) q[9];
ry(0.76525112156699) q[10];
rz(-2.2897431143429237) q[10];
ry(1.2277342056059657) q[11];
rz(1.8948183932953817) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.573169835410172) q[0];
rz(1.7053143909920028) q[0];
ry(2.2618240574619826) q[1];
rz(-1.328871486864635) q[1];
ry(1.5804050862224677) q[2];
rz(-0.007576591086852297) q[2];
ry(-1.5763312854206404) q[3];
rz(-0.0010581180808024728) q[3];
ry(-1.5685411341892976) q[4];
rz(-2.2312327273280186) q[4];
ry(-1.574777418731108) q[5];
rz(2.7580039603498294) q[5];
ry(-3.1383079626287844) q[6];
rz(2.4758298899187645) q[6];
ry(-0.006292014219379638) q[7];
rz(1.2383716380768235) q[7];
ry(-0.9141990092107799) q[8];
rz(-0.9130464327002201) q[8];
ry(1.6902090559700755) q[9];
rz(-2.612906824930084) q[9];
ry(2.307082346555205) q[10];
rz(1.0871349254544953) q[10];
ry(-2.2882349845751966) q[11];
rz(-0.7480769718154605) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.9744614296444722) q[0];
rz(2.8799669613335355) q[0];
ry(1.3232474733128168) q[1];
rz(0.236789251952477) q[1];
ry(-1.5755844876414127) q[2];
rz(-1.570901385894059) q[2];
ry(-1.5501342217390333) q[3];
rz(1.3328101820828886) q[3];
ry(-3.094060660454513) q[4];
rz(-0.668859013116383) q[4];
ry(1.5957815703345828) q[5];
rz(3.0867387448900034) q[5];
ry(2.95551357202274) q[6];
rz(3.0871709265049305) q[6];
ry(1.5703493226035359) q[7];
rz(-1.1988187666612076) q[7];
ry(1.381719400282873) q[8];
rz(-0.6034580919397916) q[8];
ry(-1.8882484570207803) q[9];
rz(2.6882198627553575) q[9];
ry(1.2913828968810614) q[10];
rz(2.3134905363230907) q[10];
ry(-1.4965671931400886) q[11];
rz(-2.7821329256515797) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.4267038748039038) q[0];
rz(1.395539972905007) q[0];
ry(-2.7844872169688832) q[1];
rz(-1.9830185201522168) q[1];
ry(-1.568861603742798) q[2];
rz(3.139713912050337) q[2];
ry(-1.5789644388735724) q[3];
rz(3.140584256213895) q[3];
ry(1.5815797543086243) q[4];
rz(0.5665269939967077) q[4];
ry(-0.03338191449269701) q[5];
rz(-1.3484195328330246) q[5];
ry(-1.5575564817397267) q[6];
rz(-1.6431046322893215) q[6];
ry(-1.5626063309312606) q[7];
rz(0.30996867302733555) q[7];
ry(1.5679893044403066) q[8];
rz(-1.5726490177900407) q[8];
ry(1.5743257934817596) q[9];
rz(-1.5572897987340162) q[9];
ry(-2.8900851697170853) q[10];
rz(2.5341218986417116) q[10];
ry(-1.093745058652553) q[11];
rz(1.0382974887004943) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.7291751566028455) q[0];
rz(1.3080453187978207) q[0];
ry(-0.6365118597980152) q[1];
rz(-2.5768872079972747) q[1];
ry(-1.5700492253637144) q[2];
rz(1.5397346983585676) q[2];
ry(1.5741819611033157) q[3];
rz(0.004684037727027857) q[3];
ry(0.007583645873156186) q[4];
rz(0.4952412686298969) q[4];
ry(3.780010531279967e-05) q[5];
rz(1.4063147909862779) q[5];
ry(-0.0006107186193520687) q[6];
rz(1.64575926868579) q[6];
ry(-0.0010270203534874511) q[7];
rz(2.829906037989579) q[7];
ry(-2.991551881493088) q[8];
rz(1.5710383574098925) q[8];
ry(-2.897917874184864) q[9];
rz(3.1348738145211064) q[9];
ry(-0.539809501820122) q[10];
rz(1.185795380467355) q[10];
ry(2.2494290478255765) q[11];
rz(0.07157725716802345) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.1378127480406737) q[0];
rz(-0.3920375599916977) q[0];
ry(2.7621922334634) q[1];
rz(-1.177423467360743) q[1];
ry(-0.2033701576951931) q[2];
rz(-0.026482813611317543) q[2];
ry(-1.5721803913196057) q[3];
rz(-1.702212866683691) q[3];
ry(-0.002401506324998505) q[4];
rz(-2.480306484954092) q[4];
ry(-1.571450290817853) q[5];
rz(-1.5683756019727735) q[5];
ry(1.5583737470697994) q[6];
rz(-0.024951604225753796) q[6];
ry(1.579736343871967) q[7];
rz(0.30591773967167407) q[7];
ry(-1.5684443820729284) q[8];
rz(1.9741334069943284) q[8];
ry(-0.04532336664673455) q[9];
rz(1.6014708519106287) q[9];
ry(2.699797925748368) q[10];
rz(2.6939599208985134) q[10];
ry(-1.8639659696021331) q[11];
rz(-1.9669907348479845) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5737139783349319) q[0];
rz(0.5583446741384321) q[0];
ry(-3.1251334780603077) q[1];
rz(1.262332382311742) q[1];
ry(-3.0883706738976144) q[2];
rz(-1.0692120677175296) q[2];
ry(-0.29581651913657403) q[3];
rz(0.017337099251380828) q[3];
ry(0.0009560426111283604) q[4];
rz(-2.733626514879571) q[4];
ry(-1.5625322744622379) q[5];
rz(-2.178190024732005) q[5];
ry(2.98738070547463) q[6];
rz(-2.6217373135870834) q[6];
ry(1.5729626936294965) q[7];
rz(1.5795981579170144) q[7];
ry(-1.5699659082840893) q[8];
rz(0.5576410865658534) q[8];
ry(0.1301304201440021) q[9];
rz(1.5671301042059547) q[9];
ry(-1.5731006919193695) q[10];
rz(-1.0151939645692603) q[10];
ry(-1.6145635073425084) q[11];
rz(1.576095255992607) q[11];