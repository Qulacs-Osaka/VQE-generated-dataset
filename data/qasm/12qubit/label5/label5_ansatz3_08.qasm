OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.6251989754131049) q[0];
rz(1.965058388880446) q[0];
ry(1.0844785672215602) q[1];
rz(2.9731306850860353) q[1];
ry(3.137908442927192) q[2];
rz(-2.74281535787747) q[2];
ry(3.1415642765585035) q[3];
rz(-0.2280143393787763) q[3];
ry(-0.0004358965881186482) q[4];
rz(-1.16439939038345) q[4];
ry(1.570913189737189) q[5];
rz(2.8387023994404896) q[5];
ry(-3.1415014373368484) q[6];
rz(-1.1803132957627296) q[6];
ry(-1.5708554974743443) q[7];
rz(2.032200068392125) q[7];
ry(-3.1411293603759507) q[8];
rz(-2.5052018045307496) q[8];
ry(0.10055648268795443) q[9];
rz(1.498858887078165) q[9];
ry(-0.6955646061082827) q[10];
rz(3.0446561200407034) q[10];
ry(-1.561595925077037) q[11];
rz(-0.00038903332958906134) q[11];
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
ry(2.643692084710378) q[0];
rz(-2.72232677296586) q[0];
ry(-2.9696902782047694) q[1];
rz(2.674284728426503) q[1];
ry(-1.3731144012715752) q[2];
rz(2.613485506995157) q[2];
ry(-0.00046329744361006675) q[3];
rz(1.2836681947171567) q[3];
ry(3.0047053404516526) q[4];
rz(-1.3810245961441492) q[4];
ry(-1.4856509177015003) q[5];
rz(1.2630237647728766) q[5];
ry(1.5708135754174433) q[6];
rz(-1.4326909132095078) q[6];
ry(-3.1288099052957765) q[7];
rz(-2.9516294189723093) q[7];
ry(-1.5744290540205468) q[8];
rz(0.00047816756952268904) q[8];
ry(-0.013645604210451613) q[9];
rz(1.6324069319429224) q[9];
ry(0.00693442205320416) q[10];
rz(0.09110312522785602) q[10];
ry(1.5566784652534422) q[11];
rz(-0.4112309547234849) q[11];
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
ry(-2.042563078735655) q[0];
rz(-0.9256421320523806) q[0];
ry(-0.63125752040105) q[1];
rz(-0.9173999513181537) q[1];
ry(-1.4802461605489903) q[2];
rz(1.3676736189095342) q[2];
ry(1.5702105943403073) q[3];
rz(1.5717168162055943) q[3];
ry(-1.699614829245428e-05) q[4];
rz(-0.7966000448399502) q[4];
ry(4.5318354855439225e-05) q[5];
rz(1.8183236562755833) q[5];
ry(0.12919283791963565) q[6];
rz(3.1393505828008674) q[6];
ry(-1.570840024824149) q[7];
rz(-0.0006323591538654809) q[7];
ry(1.570754046250928) q[8];
rz(3.1353926876329887) q[8];
ry(-2.8297197249530472) q[9];
rz(-1.570426440795929) q[9];
ry(-1.8813470705741595) q[10];
rz(-0.00012679009666527463) q[10];
ry(-0.00036603716653270087) q[11];
rz(-1.1509143445372458) q[11];
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
ry(9.593695700958217e-05) q[0];
rz(-0.48827396765718584) q[0];
ry(-1.5740176205956509) q[1];
rz(-0.31234715636531085) q[1];
ry(1.5719150499705123) q[2];
rz(2.9468979070116332) q[2];
ry(-1.571369605857125) q[3];
rz(-3.0283379912492077) q[3];
ry(3.119682480269713) q[4];
rz(-2.3403513626889816) q[4];
ry(-3.137693340296055) q[5];
rz(-0.005127599740190217) q[5];
ry(-0.332828925449004) q[6];
rz(-1.5685790835108993) q[6];
ry(-1.6284920880318676) q[7];
rz(-0.12151502105502499) q[7];
ry(-1.5717497863084928) q[8];
rz(-1.5706497387121363) q[8];
ry(1.570824816271105) q[9];
rz(-0.013064868832064964) q[9];
ry(1.5715962626118705) q[10];
rz(0.0041104481447229455) q[10];
ry(1.5704114538750709) q[11];
rz(-1.5697474846739023) q[11];
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
ry(0.823353660174163) q[0];
rz(0.108350811137508) q[0];
ry(1.6520096121863528) q[1];
rz(-2.922012917958402) q[1];
ry(1.718667381449304) q[2];
rz(-1.300708719250932) q[2];
ry(-2.5374692685472584) q[3];
rz(-2.9777423742139524) q[3];
ry(-2.924922883023136e-05) q[4];
rz(1.1478810971484639) q[4];
ry(-3.1276381942292977) q[5];
rz(-1.913590283628111) q[5];
ry(1.5723462161327533) q[6];
rz(-1.6672573859788322) q[6];
ry(-2.192661393692114) q[7];
rz(-3.08070118938466) q[7];
ry(-1.5751187171103425) q[8];
rz(1.570594213008942) q[8];
ry(0.7478759152152274) q[9];
rz(-1.2262171166689892) q[9];
ry(1.5734939123818767) q[10];
rz(1.5686141382669296) q[10];
ry(-1.5663591497263536) q[11];
rz(-7.1120068669827666e-06) q[11];
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
ry(-3.1389217144796664) q[0];
rz(2.100512349456214) q[0];
ry(-3.1377155580278315) q[1];
rz(-0.09638982332394534) q[1];
ry(3.1325793910818787) q[2];
rz(0.3057360529923985) q[2];
ry(-0.0022991048572285067) q[3];
rz(0.16452738548034448) q[3];
ry(-3.1415894716039947) q[4];
rz(3.1240910954842414) q[4];
ry(2.6897038038597995e-05) q[5];
rz(-0.060573043164525435) q[5];
ry(3.140945826486076) q[6];
rz(-1.6667780426141787) q[6];
ry(-4.8924237455061366e-05) q[7];
rz(-0.060913525108482316) q[7];
ry(0.15780534990428308) q[8];
rz(-1.570825735171923) q[8];
ry(-0.0004759203225488362) q[9];
rz(1.1474192478442722) q[9];
ry(0.8144076157157171) q[10];
rz(-0.7872407709350208) q[10];
ry(1.640936194849997) q[11];
rz(-2.337712462254134e-05) q[11];
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
ry(-3.133976194775302) q[0];
rz(2.2072582663881612) q[0];
ry(-2.363914220248629) q[1];
rz(-0.4583330965509295) q[1];
ry(-1.758837336960842) q[2];
rz(-2.991541438310306) q[2];
ry(2.8491758318783225) q[3];
rz(1.8111002172110904) q[3];
ry(-3.1415825790031455) q[4];
rz(2.0421218386304156) q[4];
ry(-0.001834578801902076) q[5];
rz(1.7708956425790658) q[5];
ry(1.5720142452042107) q[6];
rz(-3.138189254875289) q[6];
ry(2.1927087306121567) q[7];
rz(0.12072526420630682) q[7];
ry(1.5672969145519213) q[8];
rz(-1.120872928265454) q[8];
ry(1.351938392164156) q[9];
rz(3.117572947325454) q[9];
ry(-3.1413819683567046) q[10];
rz(-0.7899303400796278) q[10];
ry(1.5671678202253072) q[11];
rz(-0.03623450482710371) q[11];
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
ry(-0.0005577266110465828) q[0];
rz(-0.6348178371492598) q[0];
ry(-1.573505434540242) q[1];
rz(-2.9933638033033034) q[1];
ry(1.5749524374810373) q[2];
rz(-1.1012196430666283) q[2];
ry(1.5967263577111246) q[3];
rz(1.5713196607425894) q[3];
ry(1.478777732433285) q[4];
rz(2.792539900762159) q[4];
ry(-2.366361226999577) q[5];
rz(-2.1365705989679027) q[5];
ry(-1.579594128586532) q[6];
rz(2.6977269124322394) q[6];
ry(-2.1740045378493535) q[7];
rz(-0.41104334172255635) q[7];
ry(-3.1413913765928796) q[8];
rz(2.019896083013587) q[8];
ry(1.571084342339803) q[9];
rz(0.00029849754006061596) q[9];
ry(1.248819176922824) q[10];
rz(1.5719634890627825) q[10];
ry(-0.111798639287513) q[11];
rz(-1.5374463280156387) q[11];
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
ry(3.1405143625348613) q[0];
rz(0.3064443731859803) q[0];
ry(-0.027414152319421348) q[1];
rz(2.9917910379494805) q[1];
ry(-0.048444000857189404) q[2];
rz(-1.3088093896652984) q[2];
ry(1.5709342348027102) q[3];
rz(-1.5720149437816104) q[3];
ry(3.141592076114935) q[4];
rz(1.2198696962783202) q[4];
ry(-8.667549215836936e-05) q[5];
rz(-0.9680848383172477) q[5];
ry(-3.1415839152429283) q[6];
rz(2.257506151914077) q[6];
ry(-3.1414075490178552) q[7];
rz(0.10287085943658969) q[7];
ry(-0.1603031154186927) q[8];
rz(-1.5700253939178834) q[8];
ry(-0.5781850908579137) q[9];
rz(-1.5711535062199682) q[9];
ry(1.5704881285462922) q[10];
rz(3.0313658588456533) q[10];
ry(-1.5707734418218697) q[11];
rz(1.5677365978450988) q[11];
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
ry(-3.141527760871521) q[0];
rz(-3.0400518180082075) q[0];
ry(-1.5698631494964106) q[1];
rz(-1.570748599465697) q[1];
ry(-3.1408870401155458) q[2];
rz(-2.860386222111573) q[2];
ry(-1.5709946878918426) q[3];
rz(-3.120464969348657) q[3];
ry(-1.5907107389193078) q[4];
rz(2.900455355664842) q[4];
ry(1.9360183573629026) q[5];
rz(0.0284706953616638) q[5];
ry(-3.1323010550272214) q[6];
rz(-1.7302119094026285) q[6];
ry(3.1351957924088922) q[7];
rz(-2.3426170437816083) q[7];
ry(-1.5697536587275378) q[8];
rz(0.00010279750360812318) q[8];
ry(-1.570735705787572) q[9];
rz(-1.5708539752224018) q[9];
ry(-0.00047735628762257676) q[10];
rz(-1.4606676228768685) q[10];
ry(1.5706681762396513) q[11];
rz(-1.5709578713203474) q[11];
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
ry(3.1407944409573387) q[0];
rz(1.365850842492158) q[0];
ry(1.5649733581037646) q[1];
rz(2.427329239457126) q[1];
ry(-1.5703730247201007) q[2];
rz(-1.4631084694313239) q[2];
ry(-0.005170402088985782) q[3];
rz(-1.6968564870382536) q[3];
ry(-5.613771511381316e-05) q[4];
rz(1.7197165611060397) q[4];
ry(0.006407222166418869) q[5];
rz(1.5553155929132536) q[5];
ry(-0.015178889077493098) q[6];
rz(1.6645326385538484) q[6];
ry(-3.1359042555947085) q[7];
rz(2.506106853947279) q[7];
ry(-1.7167812965969274) q[8];
rz(-2.6496328568528615) q[8];
ry(1.5708815659662694) q[9];
rz(2.984203601356219) q[9];
ry(1.3563555644880418) q[10];
rz(-0.00239803081015176) q[10];
ry(-1.5707566992929065) q[11];
rz(-0.31143686062597786) q[11];
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
ry(-1.5710356243283663) q[0];
rz(0.6764980121329983) q[0];
ry(-3.141218342120139) q[1];
rz(-1.9299980551233749) q[1];
ry(3.1342459571862924) q[2];
rz(0.783880607038131) q[2];
ry(3.1414970947114678) q[3];
rz(0.24085998037985817) q[3];
ry(1.5708627579085057) q[4];
rz(-0.8945087750437279) q[4];
ry(-1.5369597399034867) q[5];
rz(-1.5823960090112863) q[5];
ry(1.5711767926928173) q[6];
rz(-0.8957001987057112) q[6];
ry(3.141525068239154) q[7];
rz(2.4333063980006115) q[7];
ry(3.1415851700031783) q[8];
rz(-0.4053994024490538) q[8];
ry(-0.00013821718280482287) q[9];
rz(0.37153647199284673) q[9];
ry(1.6210060797432535) q[10];
rz(-2.466266735923047) q[10];
ry(-3.141573809513606) q[11];
rz(-1.6650118570827046) q[11];