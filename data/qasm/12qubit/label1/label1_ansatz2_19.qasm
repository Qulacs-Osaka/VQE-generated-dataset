OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.0192301199050195) q[0];
rz(0.713251114063274) q[0];
ry(0.005482278944872299) q[1];
rz(0.7652048299787303) q[1];
ry(3.1108175962544315) q[2];
rz(-2.09011784193164) q[2];
ry(-2.4547685204186513) q[3];
rz(-1.3331759654648445) q[3];
ry(-2.1367610267299355) q[4];
rz(-0.20415692654079226) q[4];
ry(-1.1144323731371335) q[5];
rz(-2.6646210432178283) q[5];
ry(-0.43453448850380877) q[6];
rz(2.452076407619934) q[6];
ry(0.7409359546439616) q[7];
rz(-1.7000604223682574) q[7];
ry(-1.2802629037488575) q[8];
rz(0.7016830859858167) q[8];
ry(2.555887388054683) q[9];
rz(2.333354458332558) q[9];
ry(-2.2438120655050473) q[10];
rz(0.7989841928987574) q[10];
ry(1.9171885706357705) q[11];
rz(-0.05323121515027412) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.4734021231403949) q[0];
rz(-2.24017595740741) q[0];
ry(0.008890381578699145) q[1];
rz(-2.915210757360754) q[1];
ry(-0.02514205043112927) q[2];
rz(1.4877543148631351) q[2];
ry(2.0081933035714292) q[3];
rz(-1.313766763785436) q[3];
ry(0.7125952781058604) q[4];
rz(0.9586058359772186) q[4];
ry(0.9549724239431869) q[5];
rz(1.467895653379525) q[5];
ry(1.1290637548948315) q[6];
rz(1.7303398588816692) q[6];
ry(-0.6912354567715244) q[7];
rz(-1.9467769517589637) q[7];
ry(-2.4949987893736743) q[8];
rz(2.867367555390669) q[8];
ry(-0.5305260162279721) q[9];
rz(-1.1459064348356252) q[9];
ry(1.6308388075033662) q[10];
rz(-0.24616442170764238) q[10];
ry(2.6131994402769867) q[11];
rz(0.4505571619712993) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.764779590864343) q[0];
rz(2.3658396027285145) q[0];
ry(3.1043553242188913) q[1];
rz(-0.28938790922141683) q[1];
ry(-0.015396292617632312) q[2];
rz(-0.27362500078537616) q[2];
ry(-2.8921437177031692) q[3];
rz(-0.6294119780281101) q[3];
ry(1.720488836777355) q[4];
rz(-0.4019630453611275) q[4];
ry(-0.2757699819758752) q[5];
rz(0.09511325101829281) q[5];
ry(0.3721968982699) q[6];
rz(1.1202927562250984) q[6];
ry(2.019447745148473) q[7];
rz(-0.025444239679806854) q[7];
ry(1.7342503910304377) q[8];
rz(-2.111743382939572) q[8];
ry(1.3075167093385127) q[9];
rz(2.669835251128318) q[9];
ry(-0.5583675522542174) q[10];
rz(0.7457935840952219) q[10];
ry(-1.9658561048304322) q[11];
rz(-0.16347760513053625) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.2413457968122712) q[0];
rz(1.1607397499434917) q[0];
ry(-1.5521086142559186) q[1];
rz(0.7553516882312561) q[1];
ry(-1.544024539254246) q[2];
rz(-0.08678972692776676) q[2];
ry(-1.7718274159406837) q[3];
rz(3.1345339386967113) q[3];
ry(1.9636360250833527) q[4];
rz(0.3910055746720903) q[4];
ry(-0.7350640258136186) q[5];
rz(0.003044519908753918) q[5];
ry(2.5334601356705244) q[6];
rz(-2.536335513830235) q[6];
ry(-2.2933234974483905) q[7];
rz(-0.45896128478475173) q[7];
ry(0.9882267483007111) q[8];
rz(-2.710323197031697) q[8];
ry(2.578826714235414) q[9];
rz(-1.837624981620423) q[9];
ry(1.2882657070492547) q[10];
rz(0.05076494158981948) q[10];
ry(-2.8477346954355656) q[11];
rz(0.8812902542969994) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.1822204340362763) q[0];
rz(1.357591606776887) q[0];
ry(0.04265354682306999) q[1];
rz(-2.3319452997726913) q[1];
ry(0.19748965874878477) q[2];
rz(1.6214878170755032) q[2];
ry(1.5871715077308357) q[3];
rz(0.49484920988695) q[3];
ry(-0.9282217722882766) q[4];
rz(-0.20583887912516494) q[4];
ry(-1.9278989134522826) q[5];
rz(-2.6666907330652823) q[5];
ry(-0.9994409833003246) q[6];
rz(-1.3238314051935305) q[6];
ry(2.7894953583512736) q[7];
rz(-1.3911658226040906) q[7];
ry(2.744535866024739) q[8];
rz(3.1273410931247665) q[8];
ry(1.2032536748078702) q[9];
rz(-1.7594628244107824) q[9];
ry(0.20338222898846237) q[10];
rz(-1.570770470692453) q[10];
ry(0.8027134626599806) q[11];
rz(2.4593563736529593) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.6664273689603273) q[0];
rz(-0.47846191738599014) q[0];
ry(-1.5120009016409095) q[1];
rz(0.45879461571870533) q[1];
ry(0.5415792745098146) q[2];
rz(-1.575198506628628) q[2];
ry(-1.0311925405051119) q[3];
rz(0.5008959938534377) q[3];
ry(1.2839926306717686) q[4];
rz(0.5483118005316054) q[4];
ry(-0.31470460083841395) q[5];
rz(-1.5999368459630285) q[5];
ry(-0.2559010949571121) q[6];
rz(0.4388135793752316) q[6];
ry(-1.3850667167876394) q[7];
rz(-0.2291672502931442) q[7];
ry(-0.8843501604627911) q[8];
rz(1.214874614681423) q[8];
ry(2.703640764958334) q[9];
rz(1.1436637315853924) q[9];
ry(1.7938344998003073) q[10];
rz(-0.555357394498829) q[10];
ry(-1.7552132158508047) q[11];
rz(1.5936647995804398) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1031482555393626) q[0];
rz(0.1363792193553932) q[0];
ry(3.133912557215287) q[1];
rz(0.4348294206094257) q[1];
ry(-0.13045981653051494) q[2];
rz(-1.513610544564452) q[2];
ry(-1.929461630789274) q[3];
rz(-1.2061453046266524) q[3];
ry(-1.9972484512012898) q[4];
rz(-0.7500392444928139) q[4];
ry(1.9277266483834818) q[5];
rz(3.11098343432383) q[5];
ry(0.9760055871994955) q[6];
rz(0.1133535635588465) q[6];
ry(0.28738805951476565) q[7];
rz(1.5424833997209673) q[7];
ry(-2.6118169284578996) q[8];
rz(-2.5987455430081376) q[8];
ry(1.0006624239022055) q[9];
rz(-0.9999580178296373) q[9];
ry(3.007847721301507) q[10];
rz(-1.4900465687611992) q[10];
ry(-1.0458587910015773) q[11];
rz(1.6885097146920716) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.6669902987043761) q[0];
rz(1.2205131334101516) q[0];
ry(-2.440984161163906) q[1];
rz(-1.5207682589567337) q[1];
ry(2.7221250475341368) q[2];
rz(-1.5572795126345067) q[2];
ry(-2.8303459770954396) q[3];
rz(2.2484705422439153) q[3];
ry(2.90462459511986) q[4];
rz(2.244882160965455) q[4];
ry(1.2701169839728674) q[5];
rz(-0.24758396270950644) q[5];
ry(1.48918660677601) q[6];
rz(-2.7146325086364644) q[6];
ry(2.7787602215215923) q[7];
rz(-2.276083006263052) q[7];
ry(-1.167642283849144) q[8];
rz(0.5727672683883673) q[8];
ry(2.602638487712673) q[9];
rz(-1.5019975371633931) q[9];
ry(2.820154492395757) q[10];
rz(2.7288907147910106) q[10];
ry(1.152316899527241) q[11];
rz(2.9148478451509776) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.3744708261837144) q[0];
rz(1.7011851079646858) q[0];
ry(-0.23541831641825975) q[1];
rz(-1.6071044922315825) q[1];
ry(1.647827438834306) q[2];
rz(1.5681794070023762) q[2];
ry(-2.9888940689919705) q[3];
rz(-1.7449740989241986) q[3];
ry(-2.7504607272093815) q[4];
rz(-1.3952505060193354) q[4];
ry(-1.083944219677536) q[5];
rz(2.8712546204608445) q[5];
ry(-2.379930157573053) q[6];
rz(3.0219519915820077) q[6];
ry(1.5695163910162597) q[7];
rz(2.3127243830281072) q[7];
ry(2.6470949736673677) q[8];
rz(0.27618899634115485) q[8];
ry(0.8077258075435483) q[9];
rz(1.4219039824718962) q[9];
ry(1.5496645172487158) q[10];
rz(1.9474668843790974) q[10];
ry(-1.7704710208375838) q[11];
rz(-0.047323015498554784) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.6795816483491586) q[0];
rz(0.9936960098275807) q[0];
ry(-0.5492001151058066) q[1];
rz(0.4905026664447875) q[1];
ry(-0.9194612912696086) q[2];
rz(1.5216104037739093) q[2];
ry(-1.6375705514875258) q[3];
rz(-1.9128459774546576) q[3];
ry(-0.2984624438726773) q[4];
rz(-2.9376870717390786) q[4];
ry(-2.7663207594068226) q[5];
rz(0.11603217258263099) q[5];
ry(1.3558247825610747) q[6];
rz(1.7248430969076907) q[6];
ry(-2.5389537287237673) q[7];
rz(-1.9480244560154698) q[7];
ry(1.7686563587469701) q[8];
rz(-0.9202562125511298) q[8];
ry(-0.7539484014093065) q[9];
rz(-0.7773467251261361) q[9];
ry(-1.988187648403445) q[10];
rz(-3.037600218152459) q[10];
ry(2.66572488106128) q[11];
rz(1.6930459157553812) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.2557166286246417) q[0];
rz(-1.9070288953456058) q[0];
ry(3.110241056889571) q[1];
rz(-2.642111084119216) q[1];
ry(-0.32287164818141445) q[2];
rz(1.4055939485352846) q[2];
ry(-2.037152123075453) q[3];
rz(-1.3242158175767211) q[3];
ry(-2.87411532487279) q[4];
rz(3.117700997161054) q[4];
ry(-1.9780425412392282) q[5];
rz(-2.498396428979548) q[5];
ry(-2.455762370874011) q[6];
rz(-2.811753651073499) q[6];
ry(-0.15361559217146592) q[7];
rz(0.306042025548427) q[7];
ry(2.36935165775306) q[8];
rz(0.001227501152229846) q[8];
ry(-1.8896436520063444) q[9];
rz(1.8100447444981773) q[9];
ry(1.6667875285514828) q[10];
rz(0.5990750039827799) q[10];
ry(-1.3518891581002153) q[11];
rz(-2.732921374644805) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.9295403789095982) q[0];
rz(-2.654666538265777) q[0];
ry(1.1330262966971518) q[1];
rz(-1.5608266744728319) q[1];
ry(3.006505823301342) q[2];
rz(-1.8277085301020475) q[2];
ry(-0.8387326097924936) q[3];
rz(0.236562988899418) q[3];
ry(0.3451847081563396) q[4];
rz(0.6022757231543201) q[4];
ry(-2.508321769166282) q[5];
rz(2.7354399820231086) q[5];
ry(0.4649073133684426) q[6];
rz(-2.1564840468483926) q[6];
ry(1.687209386533631) q[7];
rz(-1.661270056093379) q[7];
ry(0.7742338712335334) q[8];
rz(2.85317379520861) q[8];
ry(1.3144389314617602) q[9];
rz(1.5599500687870282) q[9];
ry(2.154630173385323) q[10];
rz(2.95884718950458) q[10];
ry(-2.399542281451266) q[11];
rz(-2.0949849855369242) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.5819283801749644) q[0];
rz(1.4642038754871236) q[0];
ry(1.902306089725804) q[1];
rz(1.5686313421123883) q[1];
ry(-0.33055894563140303) q[2];
rz(-1.5123416868954214) q[2];
ry(2.525980254498352) q[3];
rz(1.5895477253718138) q[3];
ry(1.163908580660699) q[4];
rz(-1.7529687311647146) q[4];
ry(-1.7136010889290327) q[5];
rz(-2.1532791485881466) q[5];
ry(0.15632554720973246) q[6];
rz(0.8589995217588333) q[6];
ry(-2.34511873407873) q[7];
rz(-2.139513029309299) q[7];
ry(-2.490147836277141) q[8];
rz(1.985657540635654) q[8];
ry(-0.44300936238200406) q[9];
rz(0.20890702752751178) q[9];
ry(1.4179149509674476) q[10];
rz(-1.3677148292311956) q[10];
ry(-1.7074224941735157) q[11];
rz(1.1318589699069062) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.7199427522556148) q[0];
rz(-1.4810903175361114) q[0];
ry(0.5629133546301235) q[1];
rz(1.773810016562922) q[1];
ry(1.7476874623081757) q[2];
rz(1.5771356638047864) q[2];
ry(1.9332435800538574) q[3];
rz(3.1240516696299143) q[3];
ry(-0.28526299525005167) q[4];
rz(2.2956913281924267) q[4];
ry(1.3740384708382491) q[5];
rz(2.0379082974481815) q[5];
ry(1.0899731696684247) q[6];
rz(1.9475934345965342) q[6];
ry(2.909313094817898) q[7];
rz(-0.02087874022247327) q[7];
ry(-2.0209921754066658) q[8];
rz(-2.4306660923016143) q[8];
ry(-2.365308060998373) q[9];
rz(0.8731485680241704) q[9];
ry(-2.629899641585031) q[10];
rz(0.0342791837810994) q[10];
ry(1.4556087774446198) q[11];
rz(-0.9706002827113464) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.7950001758190686) q[0];
rz(0.19604816315971976) q[0];
ry(-3.063734532626735) q[1];
rz(1.7892541747176196) q[1];
ry(-2.6152397940231324) q[2];
rz(1.5705851336314982) q[2];
ry(0.2481598076971867) q[3];
rz(2.457212026394454) q[3];
ry(-0.8349670568657803) q[4];
rz(2.483434994863119) q[4];
ry(0.7502767891916069) q[5];
rz(1.4875099478919322) q[5];
ry(-0.7674199768922446) q[6];
rz(-0.9132728800044027) q[6];
ry(1.1693017872985711) q[7];
rz(0.0032026312790264986) q[7];
ry(1.4513297841491963) q[8];
rz(-0.34931386230566025) q[8];
ry(-1.9652733169369245) q[9];
rz(1.1649727926047646) q[9];
ry(2.875429716929213) q[10];
rz(1.0103647749811029) q[10];
ry(0.5732538446213056) q[11];
rz(-2.818814330667972) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.0604984848097665) q[0];
rz(-0.6107327406476832) q[0];
ry(2.8902580119210874) q[1];
rz(-1.3693569041419678) q[1];
ry(-2.285692890286458) q[2];
rz(1.5700317381730373) q[2];
ry(2.7803550433043553) q[3];
rz(0.24891232696476615) q[3];
ry(-1.1573843923661218) q[4];
rz(-2.9701210187670073) q[4];
ry(-2.581390895306044) q[5];
rz(1.2318387624132061) q[5];
ry(-2.2154365002423813) q[6];
rz(0.3872176154070157) q[6];
ry(0.5003944409534347) q[7];
rz(1.5189891178351245) q[7];
ry(-2.4254245661836418) q[8];
rz(0.7041919757340951) q[8];
ry(1.6328131583146155) q[9];
rz(1.6613291571195166) q[9];
ry(-0.746346874045825) q[10];
rz(-0.5452297797976753) q[10];
ry(1.4906703722086405) q[11];
rz(2.1478520132207133) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.721951342657853) q[0];
rz(-2.751626946632436) q[0];
ry(2.999175752818992) q[1];
rz(0.3387912330560336) q[1];
ry(-1.6340890738246816) q[2];
rz(1.929386409967433) q[2];
ry(0.9870709591654322) q[3];
rz(-0.11179799355352582) q[3];
ry(1.8676066208769213) q[4];
rz(-2.6800932967303903) q[4];
ry(-1.288241304043715) q[5];
rz(2.170362842382992) q[5];
ry(0.5071595280314183) q[6];
rz(-0.19964876711354676) q[6];
ry(-0.916065689140905) q[7];
rz(1.6131726433722235) q[7];
ry(-0.3427225637834379) q[8];
rz(-2.760196556749895) q[8];
ry(2.5469336768882864) q[9];
rz(1.886571959468272) q[9];
ry(0.1020154734482349) q[10];
rz(2.2037227743930075) q[10];
ry(1.1885919952674486) q[11];
rz(1.6402434827796046) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.33213030925852394) q[0];
rz(0.36236804950403645) q[0];
ry(-3.13440931673474) q[1];
rz(0.16303546083244003) q[1];
ry(-0.053249467242698384) q[2];
rz(-1.925786161688597) q[2];
ry(-1.2658701552616725) q[3];
rz(1.1996772622133234) q[3];
ry(1.439834444036636) q[4];
rz(-1.6464366484838173) q[4];
ry(0.25554104888292056) q[5];
rz(0.8318101921759445) q[5];
ry(2.550802479852462) q[6];
rz(0.9697358273043122) q[6];
ry(-2.742217113103945) q[7];
rz(2.7300277592131743) q[7];
ry(1.3366170145342737) q[8];
rz(0.18125572939599444) q[8];
ry(-2.09578354310069) q[9];
rz(0.5950747382272122) q[9];
ry(-1.2331778766976562) q[10];
rz(-1.8175350626796432) q[10];
ry(1.3306344096249916) q[11];
rz(3.0052304852730383) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.3000303824117827) q[0];
rz(2.8439550037031265) q[0];
ry(1.517856361404883) q[1];
rz(1.5572112521297001) q[1];
ry(2.352854257851911) q[2];
rz(1.5954431353466858) q[2];
ry(2.7008732597036884) q[3];
rz(0.3999690954835895) q[3];
ry(1.4030392908930305) q[4];
rz(1.6288565224879503) q[4];
ry(-1.746521944944053) q[5];
rz(3.1197472588183084) q[5];
ry(0.3903517948890264) q[6];
rz(0.3871031013787096) q[6];
ry(-2.124595927861111) q[7];
rz(2.0411723401555633) q[7];
ry(-2.60054623055293) q[8];
rz(1.4801112707839628) q[8];
ry(-0.15843208036705114) q[9];
rz(1.724665574286589) q[9];
ry(1.4198923990780932) q[10];
rz(-1.6047360521597822) q[10];
ry(-1.12047680197971) q[11];
rz(0.3852298295862786) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.48941273066908497) q[0];
rz(-0.14347903816076255) q[0];
ry(1.0197217050184333) q[1];
rz(-2.7252588986114836) q[1];
ry(0.8247554078711287) q[2];
rz(1.540765471209193) q[2];
ry(0.5280633912453386) q[3];
rz(-1.7821926689464307) q[3];
ry(-0.5998736289184512) q[4];
rz(-2.6924916967988883) q[4];
ry(-1.3907134131532377) q[5];
rz(-2.8598188495294514) q[5];
ry(1.5120427954500935) q[6];
rz(1.5501501989319273) q[6];
ry(-0.6272055179676794) q[7];
rz(2.9063209531871235) q[7];
ry(-1.597877322561118) q[8];
rz(2.0671153112105594) q[8];
ry(2.1091155295995936) q[9];
rz(2.452386456027483) q[9];
ry(-0.6077522702162449) q[10];
rz(-2.4823040109925096) q[10];
ry(-1.702120976922527) q[11];
rz(-0.7547377518562287) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.7791879612925374) q[0];
rz(-2.1074372571334843) q[0];
ry(3.1414933518239305) q[1];
rz(-2.7305658315968766) q[1];
ry(-1.43533595772466) q[2];
rz(1.5471351404775964) q[2];
ry(-2.4912684606913795) q[3];
rz(-2.6349932496088573) q[3];
ry(0.8611333465665467) q[4];
rz(-1.4194203754361912) q[4];
ry(-0.645134682920653) q[5];
rz(0.14803475727605697) q[5];
ry(1.91436275445648) q[6];
rz(2.622383745524959) q[6];
ry(-2.193882943288008) q[7];
rz(1.0039081774929546) q[7];
ry(0.6145182935014225) q[8];
rz(-0.7987211080209857) q[8];
ry(0.7740873945058695) q[9];
rz(0.6670926200973613) q[9];
ry(2.593588812332427) q[10];
rz(2.859214117835599) q[10];
ry(2.604307392392038) q[11];
rz(-3.1218875835791273) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.1374088730736631) q[0];
rz(0.6457537195651843) q[0];
ry(2.2524054072644404) q[1];
rz(-1.2370062319270456) q[1];
ry(0.55530740346605) q[2];
rz(-0.1765660414407567) q[2];
ry(0.17706055684819422) q[3];
rz(2.5903019547726) q[3];
ry(0.15474873254257007) q[4];
rz(-2.192707995967566) q[4];
ry(-1.424888673491779) q[5];
rz(0.5482019429171653) q[5];
ry(-2.0334034396686906) q[6];
rz(0.8911283313092059) q[6];
ry(-1.8834964742199922) q[7];
rz(1.8343570095669524) q[7];
ry(0.8781679597771856) q[8];
rz(-0.4505864067848835) q[8];
ry(2.5075802907851426) q[9];
rz(-1.6302511639545687) q[9];
ry(2.683731621030125) q[10];
rz(-0.38338168433157893) q[10];
ry(0.8380570453061877) q[11];
rz(-1.0734722098436276) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.1965387457426933) q[0];
rz(-2.6316314116804644) q[0];
ry(1.5133505796336904) q[1];
rz(2.024291115264334) q[1];
ry(-0.29689066776196515) q[2];
rz(-2.4624480989515223) q[2];
ry(2.7240358854512166) q[3];
rz(0.7988795964072225) q[3];
ry(-1.9899243170560577) q[4];
rz(-1.3076890121062124) q[4];
ry(-1.2403892037075817) q[5];
rz(2.4889598578140935) q[5];
ry(-2.645866696626754) q[6];
rz(0.07545256806411961) q[6];
ry(2.642223355696225) q[7];
rz(-2.030007425999856) q[7];
ry(-2.95039775479807) q[8];
rz(0.9848363401792248) q[8];
ry(-1.252257083747942) q[9];
rz(-0.025015937693645185) q[9];
ry(2.129942897856104) q[10];
rz(0.803720873558416) q[10];
ry(1.465598229869102) q[11];
rz(2.290033683551882) q[11];