OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.6956733075525525) q[0];
rz(-2.1177913408424063) q[0];
ry(2.289050424740538) q[1];
rz(-1.011666236686456) q[1];
ry(0.35701422133300126) q[2];
rz(2.230202721795984) q[2];
ry(-0.4473633625835664) q[3];
rz(0.1149058170818656) q[3];
ry(1.9873146642372816) q[4];
rz(-0.14292421726482019) q[4];
ry(0.07749173513545027) q[5];
rz(-2.1103176296675645) q[5];
ry(2.2213119400076353) q[6];
rz(-0.18955502117012377) q[6];
ry(-0.5323443685398955) q[7];
rz(0.3020691636635513) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.4675218833746406) q[0];
rz(1.0384933288181835) q[0];
ry(2.4639919978507097) q[1];
rz(-0.35480816627683787) q[1];
ry(-1.6789359142455822) q[2];
rz(-2.9700801969609145) q[2];
ry(-0.26324516879737825) q[3];
rz(-1.2709797791300526) q[3];
ry(1.2148595229757957) q[4];
rz(1.4749314838700385) q[4];
ry(-2.829362671744093) q[5];
rz(2.3137533631276206) q[5];
ry(-2.2157471889300537) q[6];
rz(2.360473514116327) q[6];
ry(0.18212242145943341) q[7];
rz(-3.056773173494458) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.4745453866670433) q[0];
rz(-1.622987951553024) q[0];
ry(-2.4481712912605142) q[1];
rz(-0.6692342484909186) q[1];
ry(-2.2697968715214163) q[2];
rz(1.5965079357873087) q[2];
ry(-0.3020378126830929) q[3];
rz(-1.2835188201403271) q[3];
ry(1.6304015846546767) q[4];
rz(-1.3229683235600347) q[4];
ry(-2.7017753185191364) q[5];
rz(-2.934246754910691) q[5];
ry(-0.19388485796621577) q[6];
rz(1.0976785728485414) q[6];
ry(-2.9165148987795724) q[7];
rz(-2.7041184086701002) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.347884066556807) q[0];
rz(-2.9206455756660064) q[0];
ry(1.6565172067208322) q[1];
rz(-0.21124716093075752) q[1];
ry(0.20225512954261238) q[2];
rz(-0.0858377847348697) q[2];
ry(-0.9056315464783601) q[3];
rz(-2.0543615244940456) q[3];
ry(-2.869485120260752) q[4];
rz(0.7585705140441147) q[4];
ry(1.8193599710912043) q[5];
rz(-1.5887789068782459) q[5];
ry(0.8424912456659738) q[6];
rz(2.8635225483803293) q[6];
ry(-2.0222087193046567) q[7];
rz(-2.5327158717553817) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.8468376180611532) q[0];
rz(-1.3574918688326796) q[0];
ry(2.6427933457897748) q[1];
rz(2.5185140051114745) q[1];
ry(1.4181571842605232) q[2];
rz(0.14982226957917855) q[2];
ry(1.4025248409686863) q[3];
rz(3.005613726365369) q[3];
ry(-1.3688431868790425) q[4];
rz(-2.3546157648794286) q[4];
ry(-2.8675237011518844) q[5];
rz(-1.2590956063285654) q[5];
ry(2.4429107321747) q[6];
rz(0.2617398806311231) q[6];
ry(-2.842652158286551) q[7];
rz(1.9655562137282578) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.4479080641867) q[0];
rz(-2.020506480921223) q[0];
ry(-2.1318085684357473) q[1];
rz(-3.0153516899496453) q[1];
ry(-1.2948867933544965) q[2];
rz(-1.2156469162475416) q[2];
ry(2.6381285530815255) q[3];
rz(2.7393846536754842) q[3];
ry(-0.009802631447385579) q[4];
rz(-0.414250357638231) q[4];
ry(1.6763506282900622) q[5];
rz(1.143554322269076) q[5];
ry(1.5250486781033512) q[6];
rz(1.504510140363858) q[6];
ry(-1.4656600050360933) q[7];
rz(1.3950085566991899) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.3308824748060732) q[0];
rz(0.9128973934911042) q[0];
ry(2.786070306278334) q[1];
rz(-2.1968759684085555) q[1];
ry(1.6059255469050366) q[2];
rz(-0.03542454713636466) q[2];
ry(2.623505184765983) q[3];
rz(0.3137527375558191) q[3];
ry(-0.9764719189682547) q[4];
rz(-1.220967735149469) q[4];
ry(1.0381378111276969) q[5];
rz(-2.591033409072269) q[5];
ry(-2.7563091413304144) q[6];
rz(-2.300928698183851) q[6];
ry(-1.8115571676394353) q[7];
rz(0.0942389955465868) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7616553909302226) q[0];
rz(-1.595526414666639) q[0];
ry(-1.1563663947963612) q[1];
rz(2.977320376052207) q[1];
ry(-2.7344562225114) q[2];
rz(2.730519145892199) q[2];
ry(-1.2478392533985598) q[3];
rz(1.16982218426448) q[3];
ry(-2.1309146158252217) q[4];
rz(-1.1119575187304107) q[4];
ry(-1.690531101148972) q[5];
rz(1.1505999600413412) q[5];
ry(-2.5681676694987616) q[6];
rz(0.508023813477549) q[6];
ry(2.323341476167044) q[7];
rz(0.9812746985245079) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.3304963198331103) q[0];
rz(1.8362544373426308) q[0];
ry(-1.1059807874594911) q[1];
rz(-0.3461788830661567) q[1];
ry(0.3011940452143325) q[2];
rz(1.4271279020867604) q[2];
ry(1.073675831717735) q[3];
rz(-2.197583542151569) q[3];
ry(-0.8799345445378428) q[4];
rz(2.071916900708114) q[4];
ry(2.2951548267636865) q[5];
rz(0.6723175622441315) q[5];
ry(0.8995669035324667) q[6];
rz(2.6238338937929435) q[6];
ry(-0.33019654314654073) q[7];
rz(-0.8231370228533202) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.62470833487596) q[0];
rz(3.0061637601262294) q[0];
ry(-2.030939504850216) q[1];
rz(-0.702342956989944) q[1];
ry(-0.38370632491781453) q[2];
rz(-1.4288607456754594) q[2];
ry(1.140012181846924) q[3];
rz(-0.8239647061818349) q[3];
ry(-1.3658897531226744) q[4];
rz(0.10730239137698594) q[4];
ry(2.573662995859475) q[5];
rz(-0.14167133289803147) q[5];
ry(1.6493196822686718) q[6];
rz(-2.5925842466032933) q[6];
ry(0.45956451670332044) q[7];
rz(1.9961508287787044) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.0237737037787813) q[0];
rz(1.0001535119588316) q[0];
ry(-1.0392024129635433) q[1];
rz(-2.8151633363199484) q[1];
ry(2.60037189506057) q[2];
rz(1.5906335149783386) q[2];
ry(-2.070441183966608) q[3];
rz(-3.094855259077953) q[3];
ry(2.9746466017618007) q[4];
rz(1.4220006151142384) q[4];
ry(1.2647922960670799) q[5];
rz(-0.8244273989036018) q[5];
ry(1.2520745282939985) q[6];
rz(-2.03951682863668) q[6];
ry(-1.5409106950744302) q[7];
rz(2.212206950945575) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.8725711141658055) q[0];
rz(-3.0384801477080265) q[0];
ry(-1.2726804761651076) q[1];
rz(-0.4243950645684036) q[1];
ry(1.4111712567691077) q[2];
rz(-2.3707499652314117) q[2];
ry(1.0415606853956332) q[3];
rz(2.8026295241254506) q[3];
ry(-0.9544282892210122) q[4];
rz(0.5103381861904221) q[4];
ry(-2.3956644921888137) q[5];
rz(-0.6067391209534411) q[5];
ry(-1.2451772616329155) q[6];
rz(-0.4679359709978997) q[6];
ry(-1.7023176429342861) q[7];
rz(1.1354346933597783) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.37985911323207894) q[0];
rz(-0.05565120511255601) q[0];
ry(1.8551213114625043) q[1];
rz(3.120165803131642) q[1];
ry(0.3884759601377956) q[2];
rz(1.855450604588353) q[2];
ry(-2.6695714596292466) q[3];
rz(-2.4774900909686264) q[3];
ry(2.1095813682615816) q[4];
rz(0.7087936867593455) q[4];
ry(2.8468134528146174) q[5];
rz(0.7129298281964163) q[5];
ry(2.1837186941336464) q[6];
rz(-0.5696462160312986) q[6];
ry(1.3566510880067462) q[7];
rz(0.691137862151491) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.3229174040600495) q[0];
rz(-0.36713289307755304) q[0];
ry(-2.6828423417184273) q[1];
rz(-2.6372631939594164) q[1];
ry(-1.7991710015769247) q[2];
rz(-1.0316938715925756) q[2];
ry(1.9632418372923697) q[3];
rz(-2.0476238729721508) q[3];
ry(-2.9014325686855176) q[4];
rz(1.3976510061669591) q[4];
ry(1.697308951349502) q[5];
rz(-0.8419835044748537) q[5];
ry(-0.08219658909767169) q[6];
rz(-2.169159222091614) q[6];
ry(-2.2698247800139226) q[7];
rz(-2.807846932676435) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.447839466988305) q[0];
rz(-1.562736219849101) q[0];
ry(0.1929617097037389) q[1];
rz(2.509682219396678) q[1];
ry(-1.9436656274465272) q[2];
rz(1.6563271554552925) q[2];
ry(1.128816219691915) q[3];
rz(1.3112901292353263) q[3];
ry(-1.6343107484116288) q[4];
rz(0.6554290807505636) q[4];
ry(-2.7587947671586632) q[5];
rz(1.9094776468593118) q[5];
ry(2.123409216085249) q[6];
rz(1.6138947243835897) q[6];
ry(2.5008788834906928) q[7];
rz(-0.5963392991718154) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.2796643887142611) q[0];
rz(0.32442119716099865) q[0];
ry(2.32847996122831) q[1];
rz(1.8648823626710964) q[1];
ry(-1.210114935316005) q[2];
rz(-0.67201442974955) q[2];
ry(2.12968277196653) q[3];
rz(1.9182136534844565) q[3];
ry(2.2165060041019684) q[4];
rz(-2.873391186137222) q[4];
ry(0.7676765556884213) q[5];
rz(-2.247759558583534) q[5];
ry(1.6317959423774122) q[6];
rz(2.1048634755772824) q[6];
ry(2.284136899986882) q[7];
rz(0.2612593777010046) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.93270167843154) q[0];
rz(-0.14694739089467865) q[0];
ry(-0.36390053194225125) q[1];
rz(2.298209878195116) q[1];
ry(-0.1345194045384588) q[2];
rz(-0.48959453358501115) q[2];
ry(-1.6878001116340622) q[3];
rz(3.1284971089529217) q[3];
ry(-2.875311005571468) q[4];
rz(-2.114621057396608) q[4];
ry(-1.0228581683795748) q[5];
rz(1.3411328745710192) q[5];
ry(2.8419102926301325) q[6];
rz(-2.3629742595877854) q[6];
ry(1.7421125024336777) q[7];
rz(0.09029706285530105) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.37139895983359317) q[0];
rz(0.4573565448735385) q[0];
ry(-2.5359033017679606) q[1];
rz(-0.9349427432713462) q[1];
ry(2.2733309139629148) q[2];
rz(2.7015834823161535) q[2];
ry(-0.9943659642675061) q[3];
rz(-1.4751043186913535) q[3];
ry(-2.6223422914541286) q[4];
rz(-2.5354162403235687) q[4];
ry(2.608479812567702) q[5];
rz(-2.330757488061802) q[5];
ry(-1.3190908826918877) q[6];
rz(1.940769688964873) q[6];
ry(-2.9068152254809023) q[7];
rz(-1.2457149743436826) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.779150271522207) q[0];
rz(-1.5155043625477427) q[0];
ry(1.1145703914911609) q[1];
rz(-0.6470645097707912) q[1];
ry(0.48084241848015186) q[2];
rz(-2.3487104898780844) q[2];
ry(-0.3056937117821077) q[3];
rz(-0.5062407599916222) q[3];
ry(-1.2549302651694416) q[4];
rz(0.24580308564662887) q[4];
ry(2.555390354784448) q[5];
rz(0.7287218448916066) q[5];
ry(1.2142064915153892) q[6];
rz(-2.355648413837788) q[6];
ry(0.19646406944641168) q[7];
rz(-2.978751646337991) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.0901071412438608) q[0];
rz(-0.12100665143547382) q[0];
ry(-0.9839162616701211) q[1];
rz(1.9324712841693272) q[1];
ry(-3.02241517490441) q[2];
rz(-0.21262026241263324) q[2];
ry(-0.9581518335551049) q[3];
rz(2.4854494634188398) q[3];
ry(2.617552086171896) q[4];
rz(-2.3117318027927376) q[4];
ry(-1.1413167869385565) q[5];
rz(-0.2861404373357788) q[5];
ry(1.3798340535200948) q[6];
rz(-0.8712333517948752) q[6];
ry(1.6404552465772753) q[7];
rz(-2.494495267448409) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.9511043806414214) q[0];
rz(-3.105153197083229) q[0];
ry(-2.095172705768449) q[1];
rz(-0.9164434923798148) q[1];
ry(-0.8547103309381042) q[2];
rz(1.2309683149704673) q[2];
ry(2.1989251791347435) q[3];
rz(1.8491291746882945) q[3];
ry(0.0747510023566349) q[4];
rz(-3.01125793472955) q[4];
ry(-1.8550807000556426) q[5];
rz(0.22540827097601568) q[5];
ry(-2.352283675412106) q[6];
rz(1.4629869686557941) q[6];
ry(-2.622437261343514) q[7];
rz(2.3510054399844993) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.500556829367162) q[0];
rz(2.5276903388426613) q[0];
ry(-2.3362735478930046) q[1];
rz(2.089038183829726) q[1];
ry(2.1722517811249933) q[2];
rz(2.803382677601895) q[2];
ry(1.115505342507876) q[3];
rz(-0.3553859878975949) q[3];
ry(2.3480113185163276) q[4];
rz(1.0029863201266656) q[4];
ry(1.4756034285849116) q[5];
rz(-3.1193800384699863) q[5];
ry(1.7526459221434152) q[6];
rz(2.4761853697280256) q[6];
ry(-1.594357063974458) q[7];
rz(1.5684753334109267) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7757395008057053) q[0];
rz(-2.101627483228724) q[0];
ry(1.873494711305396) q[1];
rz(-1.6261864402925248) q[1];
ry(-1.0204876461202703) q[2];
rz(-1.4361000653856806) q[2];
ry(-2.6791323779431564) q[3];
rz(2.794996996421628) q[3];
ry(1.647291000840831) q[4];
rz(1.133397424553638) q[4];
ry(1.531357804431588) q[5];
rz(-0.3524349860854793) q[5];
ry(0.8484251520894439) q[6];
rz(0.07753065289936403) q[6];
ry(0.20310823455258387) q[7];
rz(1.5177308929335906) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.32585504983837094) q[0];
rz(-2.388592066845207) q[0];
ry(0.21958417341487027) q[1];
rz(2.499908395524523) q[1];
ry(-2.653661162628683) q[2];
rz(-2.7037568172114916) q[2];
ry(0.16095328706301928) q[3];
rz(0.38873049617579714) q[3];
ry(-1.2616563898154702) q[4];
rz(-1.841952108316112) q[4];
ry(-1.8808062153953586) q[5];
rz(-2.569244249983104) q[5];
ry(-1.3855520224399318) q[6];
rz(2.583590232605694) q[6];
ry(0.1796859821230878) q[7];
rz(2.0827417631616436) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.02796167985308) q[0];
rz(2.1077995078018033) q[0];
ry(-1.2349489151273283) q[1];
rz(2.3948542346616293) q[1];
ry(-2.0699008567315) q[2];
rz(1.6035509143746127) q[2];
ry(2.7928039593132947) q[3];
rz(1.90265152553465) q[3];
ry(0.5056910244698765) q[4];
rz(0.4364004122721248) q[4];
ry(0.38679268976306924) q[5];
rz(2.9540465040996824) q[5];
ry(-2.388611467902735) q[6];
rz(0.46953830781222816) q[6];
ry(-2.406734447332525) q[7];
rz(-1.681059925682626) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7840362363248278) q[0];
rz(-1.6853924816243788) q[0];
ry(-2.0469941412746953) q[1];
rz(0.41169704614491937) q[1];
ry(-0.721658609138701) q[2];
rz(-1.9505651765738705) q[2];
ry(-2.2433451765988326) q[3];
rz(-2.111678733701919) q[3];
ry(2.5247984108504635) q[4];
rz(-2.6484110816875077) q[4];
ry(1.429532272754322) q[5];
rz(-1.2536953873475911) q[5];
ry(-2.0083228420678423) q[6];
rz(0.1457533404446208) q[6];
ry(-0.6432870638737773) q[7];
rz(0.9758891531473726) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.9198459727519763) q[0];
rz(-2.037137891613971) q[0];
ry(1.2676310588912318) q[1];
rz(0.23455574988986144) q[1];
ry(1.741653626509316) q[2];
rz(-2.652048517452044) q[2];
ry(-2.4149710347336595) q[3];
rz(3.0247712700802465) q[3];
ry(-2.926572018011015) q[4];
rz(0.7466676828405482) q[4];
ry(1.7349153012295915) q[5];
rz(0.5739536264822097) q[5];
ry(1.0439461815468611) q[6];
rz(-0.5594528060262762) q[6];
ry(2.5217421057105556) q[7];
rz(-0.25185629726448955) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.6392966066285952) q[0];
rz(2.4780968881376495) q[0];
ry(2.837419992440192) q[1];
rz(2.82070560391752) q[1];
ry(-1.4784668159367884) q[2];
rz(2.4337611363834166) q[2];
ry(2.2976446768776233) q[3];
rz(1.910085680277521) q[3];
ry(1.955881378965544) q[4];
rz(-0.6621112170747505) q[4];
ry(2.7493009067121212) q[5];
rz(0.4567240862798032) q[5];
ry(-1.2456311358629168) q[6];
rz(-2.873634654104538) q[6];
ry(0.6677985231372903) q[7];
rz(2.0911523295051264) q[7];