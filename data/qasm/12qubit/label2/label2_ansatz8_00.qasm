OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.0581912604351764) q[0];
ry(-2.2356914548236126) q[1];
cx q[0],q[1];
ry(0.9332091956565838) q[0];
ry(1.034569387270106) q[1];
cx q[0],q[1];
ry(2.1674092646531538) q[2];
ry(3.1104762353075706) q[3];
cx q[2],q[3];
ry(3.0122175019217434) q[2];
ry(0.7005805373651705) q[3];
cx q[2],q[3];
ry(-0.9991518691831462) q[4];
ry(0.9119225481964646) q[5];
cx q[4],q[5];
ry(-1.9112434724147602) q[4];
ry(-2.0189859584377157) q[5];
cx q[4],q[5];
ry(-2.3254301515677507) q[6];
ry(1.9237185507079735) q[7];
cx q[6],q[7];
ry(2.765792575233005) q[6];
ry(1.490979519057701) q[7];
cx q[6],q[7];
ry(3.0054839838384355) q[8];
ry(-1.592388932425251) q[9];
cx q[8],q[9];
ry(0.28025915767776266) q[8];
ry(-3.012935263180335) q[9];
cx q[8],q[9];
ry(0.010835041848785032) q[10];
ry(2.7197881537122655) q[11];
cx q[10],q[11];
ry(1.9936933251739675) q[10];
ry(2.300527296118508) q[11];
cx q[10],q[11];
ry(-2.1714287707182303) q[0];
ry(2.9610308219480204) q[2];
cx q[0],q[2];
ry(-3.1018508736980697) q[0];
ry(-2.996868597367672) q[2];
cx q[0],q[2];
ry(1.628931673505761) q[2];
ry(1.448781180373308) q[4];
cx q[2],q[4];
ry(0.018055428854785838) q[2];
ry(0.007303141550882408) q[4];
cx q[2],q[4];
ry(-0.8213800652895813) q[4];
ry(-1.2992268051834939) q[6];
cx q[4],q[6];
ry(3.03842551499349) q[4];
ry(0.19025032725975866) q[6];
cx q[4],q[6];
ry(-0.5432378808733349) q[6];
ry(2.8059436670103346) q[8];
cx q[6],q[8];
ry(-0.04644857534784563) q[6];
ry(-3.14147092030045) q[8];
cx q[6],q[8];
ry(-1.8037105710825543) q[8];
ry(-2.6175818929290586) q[10];
cx q[8],q[10];
ry(-2.930156015800214) q[8];
ry(3.08537504787752) q[10];
cx q[8],q[10];
ry(1.352935497717569) q[1];
ry(2.072639385311539) q[3];
cx q[1],q[3];
ry(-2.985797514478821) q[1];
ry(1.865406213995512) q[3];
cx q[1],q[3];
ry(1.417117374882183) q[3];
ry(1.374808267540611) q[5];
cx q[3],q[5];
ry(-3.140784957567333) q[3];
ry(-3.1407435158933152) q[5];
cx q[3],q[5];
ry(2.031158155670494) q[5];
ry(2.2705358760615737) q[7];
cx q[5],q[7];
ry(-3.106868606205262) q[5];
ry(2.87713165621115) q[7];
cx q[5],q[7];
ry(0.1855841864973539) q[7];
ry(0.14619172294621308) q[9];
cx q[7],q[9];
ry(-1.9403552275073093) q[7];
ry(0.0050803114438727195) q[9];
cx q[7],q[9];
ry(-0.9492477122988605) q[9];
ry(0.22146450882005203) q[11];
cx q[9],q[11];
ry(1.5913949434180141) q[9];
ry(-3.140380260297136) q[11];
cx q[9],q[11];
ry(-2.6572790014125003) q[0];
ry(-2.5790081538335663) q[1];
cx q[0],q[1];
ry(-0.10414265794632803) q[0];
ry(2.023978091008579) q[1];
cx q[0],q[1];
ry(-0.20799089185760525) q[2];
ry(2.1663102453367253) q[3];
cx q[2],q[3];
ry(-1.377845345195918) q[2];
ry(3.121381625589472) q[3];
cx q[2],q[3];
ry(2.2264727670346076) q[4];
ry(-1.431386903312056) q[5];
cx q[4],q[5];
ry(-2.415474038520344) q[4];
ry(1.8013517113986381) q[5];
cx q[4],q[5];
ry(-0.6609360687015691) q[6];
ry(-2.6256206166781966) q[7];
cx q[6],q[7];
ry(-2.712467395910414) q[6];
ry(-1.5549912546672386) q[7];
cx q[6],q[7];
ry(-1.8247550107067272) q[8];
ry(-0.8649427205869431) q[9];
cx q[8],q[9];
ry(-3.1209119152519693) q[8];
ry(1.5544091163832714) q[9];
cx q[8],q[9];
ry(-0.18930628549965767) q[10];
ry(2.8830184556962624) q[11];
cx q[10],q[11];
ry(-1.049692800484653) q[10];
ry(0.7883747266957659) q[11];
cx q[10],q[11];
ry(-0.9857182470938496) q[0];
ry(-2.8999572194739915) q[2];
cx q[0],q[2];
ry(-3.1318321425953353) q[0];
ry(-1.2848499442473633) q[2];
cx q[0],q[2];
ry(-2.504790936472925) q[2];
ry(1.83749400333662) q[4];
cx q[2],q[4];
ry(2.986571851243222) q[2];
ry(-3.131528613487126) q[4];
cx q[2],q[4];
ry(-0.13437317988269268) q[4];
ry(-1.667574201763396) q[6];
cx q[4],q[6];
ry(-3.096127932096032) q[4];
ry(-3.087184539920418) q[6];
cx q[4],q[6];
ry(-3.0492756575904165) q[6];
ry(-0.7192192033802728) q[8];
cx q[6],q[8];
ry(0.0019410851267123164) q[6];
ry(-3.115695181527611) q[8];
cx q[6],q[8];
ry(1.694328451995416) q[8];
ry(-0.7523333092369224) q[10];
cx q[8],q[10];
ry(3.131922503205987) q[8];
ry(0.0022284511965153797) q[10];
cx q[8],q[10];
ry(1.5153664745921824) q[1];
ry(-0.00696530115703111) q[3];
cx q[1],q[3];
ry(1.3129735705505061) q[1];
ry(1.8899478190363084) q[3];
cx q[1],q[3];
ry(2.9702944138898104) q[3];
ry(0.6814101257285969) q[5];
cx q[3],q[5];
ry(-0.054196785964808675) q[3];
ry(-3.084448504972146) q[5];
cx q[3],q[5];
ry(-2.100178117773135) q[5];
ry(-0.06285022152077682) q[7];
cx q[5],q[7];
ry(-3.127699895618816) q[5];
ry(0.03984747940372904) q[7];
cx q[5],q[7];
ry(0.21398465820636492) q[7];
ry(1.549462715128513) q[9];
cx q[7],q[9];
ry(-0.011626563252328454) q[7];
ry(3.090836175774206) q[9];
cx q[7],q[9];
ry(-1.2567280770982343) q[9];
ry(-2.8244659420262974) q[11];
cx q[9],q[11];
ry(-0.3481199647125743) q[9];
ry(2.9747620762379823) q[11];
cx q[9],q[11];
ry(1.6092503208116358) q[0];
ry(-0.6292920545057754) q[1];
cx q[0],q[1];
ry(-0.9778237431629204) q[0];
ry(-2.8765557311403382) q[1];
cx q[0],q[1];
ry(-1.9694420839677749) q[2];
ry(0.287593298107395) q[3];
cx q[2],q[3];
ry(2.938425579893683) q[2];
ry(2.873913308323359) q[3];
cx q[2],q[3];
ry(2.97028489864231) q[4];
ry(1.7192781841128042) q[5];
cx q[4],q[5];
ry(-1.6044381253032258) q[4];
ry(-1.636923275632075) q[5];
cx q[4],q[5];
ry(-2.414008329360488) q[6];
ry(2.8725341400426205) q[7];
cx q[6],q[7];
ry(0.019556578084922897) q[6];
ry(-3.111507509058075) q[7];
cx q[6],q[7];
ry(1.785852028918487) q[8];
ry(1.5733136513147383) q[9];
cx q[8],q[9];
ry(-1.2575232278389483) q[8];
ry(0.0432342984363876) q[9];
cx q[8],q[9];
ry(0.7590618907994554) q[10];
ry(2.79925330225382) q[11];
cx q[10],q[11];
ry(-0.39758592773359336) q[10];
ry(2.040238256163611) q[11];
cx q[10],q[11];
ry(0.28071085378059907) q[0];
ry(1.345293337776653) q[2];
cx q[0],q[2];
ry(-3.1249687791239382) q[0];
ry(3.126422580546509) q[2];
cx q[0],q[2];
ry(-0.43303652131571163) q[2];
ry(-0.009283252213026394) q[4];
cx q[2],q[4];
ry(-0.16731478970745295) q[2];
ry(3.1409904190362563) q[4];
cx q[2],q[4];
ry(-1.068263012044568) q[4];
ry(0.6229214901383311) q[6];
cx q[4],q[6];
ry(3.138978361612869) q[4];
ry(0.010316363727772782) q[6];
cx q[4],q[6];
ry(0.5243663025177168) q[6];
ry(-1.975153111114052) q[8];
cx q[6],q[8];
ry(0.0019073798153863566) q[6];
ry(-0.03283561870393344) q[8];
cx q[6],q[8];
ry(2.032078615475963) q[8];
ry(1.8849405393057375) q[10];
cx q[8],q[10];
ry(0.019390308061463593) q[8];
ry(-3.1384892886934543) q[10];
cx q[8],q[10];
ry(-2.432015449763369) q[1];
ry(-1.305834753671741) q[3];
cx q[1],q[3];
ry(-3.132579148003903) q[1];
ry(3.1383561314442443) q[3];
cx q[1],q[3];
ry(0.07848378571858357) q[3];
ry(-1.184371654047319) q[5];
cx q[3],q[5];
ry(0.011856179749575979) q[3];
ry(0.0009086465391407091) q[5];
cx q[3],q[5];
ry(2.495251385172887) q[5];
ry(2.8275815722061957) q[7];
cx q[5],q[7];
ry(0.000681217287975322) q[5];
ry(-0.004741946772268581) q[7];
cx q[5],q[7];
ry(-2.7127020970628415) q[7];
ry(2.572482972088002) q[9];
cx q[7],q[9];
ry(-0.006760133411650693) q[7];
ry(3.124356004904497) q[9];
cx q[7],q[9];
ry(2.140566476723153) q[9];
ry(-2.350031699388922) q[11];
cx q[9],q[11];
ry(0.00277577646412297) q[9];
ry(2.994851036344745) q[11];
cx q[9],q[11];
ry(1.232500420586028) q[0];
ry(2.911480067956368) q[1];
ry(2.50393951364821) q[2];
ry(1.072753155286505) q[3];
ry(-2.5279776977259028) q[4];
ry(-0.19922500959102704) q[5];
ry(0.594292412564168) q[6];
ry(0.8470577013923478) q[7];
ry(2.3181513094786927) q[8];
ry(-2.0811057489734517) q[9];
ry(-1.6788791881548581) q[10];
ry(1.7660395446363015) q[11];