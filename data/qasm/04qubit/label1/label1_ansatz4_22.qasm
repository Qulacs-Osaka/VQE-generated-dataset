OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.5413311149064644) q[0];
rz(-0.7186590660992664) q[0];
ry(2.551467617721855) q[1];
rz(-2.7271834008872506) q[1];
ry(-0.5466385965915309) q[2];
rz(1.2834762893556306) q[2];
ry(-2.5847691651222076) q[3];
rz(-2.7202776986417865) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6256950772958011) q[0];
rz(0.7265916078975759) q[0];
ry(-2.862841335906493) q[1];
rz(0.5512395677449318) q[1];
ry(1.7634647089611435) q[2];
rz(1.411940876764745) q[2];
ry(-1.754119648635145) q[3];
rz(-0.24878616671659362) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.08857826310483129) q[0];
rz(-1.7314084940004282) q[0];
ry(-2.27706382971826) q[1];
rz(-1.2990434219175013) q[1];
ry(2.388174128753431) q[2];
rz(-0.12382305981219316) q[2];
ry(2.8955921621951672) q[3];
rz(2.4489539344030122) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.7317409492164044) q[0];
rz(-0.8951030715306137) q[0];
ry(-2.5274389360462304) q[1];
rz(2.1946089872690715) q[1];
ry(1.2462693095998933) q[2];
rz(-0.796202742787783) q[2];
ry(-2.5232266935454843) q[3];
rz(-1.6597792595402447) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.934739423946777) q[0];
rz(0.7089414487729151) q[0];
ry(2.3359064233576805) q[1];
rz(-0.3540217521527743) q[1];
ry(-1.833261886246332) q[2];
rz(1.5962368910401268) q[2];
ry(1.5582275355866495) q[3];
rz(-2.469006423629941) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.003967545488374908) q[0];
rz(1.865420759142104) q[0];
ry(-0.12962103637443523) q[1];
rz(0.6115397582969203) q[1];
ry(-2.5009334813833113) q[2];
rz(-0.9644212609657936) q[2];
ry(-0.6441304949083371) q[3];
rz(-0.024514603674148747) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7098625517364328) q[0];
rz(1.7257708925887167) q[0];
ry(-0.09461057974758684) q[1];
rz(0.7579449502721634) q[1];
ry(1.750129934290335) q[2];
rz(-1.1948823555311707) q[2];
ry(0.816814815970365) q[3];
rz(0.8327599073225818) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1597737424162398) q[0];
rz(-2.755283739015272) q[0];
ry(-1.7931821267206116) q[1];
rz(0.918771849947023) q[1];
ry(-1.587464933259464) q[2];
rz(0.27527892841928203) q[2];
ry(2.8012669424353542) q[3];
rz(-2.83183559047185) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6544405604816601) q[0];
rz(-2.0910408329765753) q[0];
ry(1.2507843508446301) q[1];
rz(-0.210898947068129) q[1];
ry(0.5692028854708314) q[2];
rz(-2.393956560739662) q[2];
ry(-2.643118136526129) q[3];
rz(1.2240652429274235) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.998561044323109) q[0];
rz(0.7970548678807401) q[0];
ry(1.1045671735793252) q[1];
rz(-0.13838880272895168) q[1];
ry(2.75144211671331) q[2];
rz(2.63251699084565) q[2];
ry(1.5273549541464106) q[3];
rz(-2.8975398189996495) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.3117889826745213) q[0];
rz(2.7055434603644093) q[0];
ry(-1.9144152270679342) q[1];
rz(0.6054967553578223) q[1];
ry(-0.023446600276802638) q[2];
rz(-2.00488695358193) q[2];
ry(-1.215415590621241) q[3];
rz(-0.5106848568749384) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5981295539118577) q[0];
rz(-2.0336674756935245) q[0];
ry(2.9295771067708816) q[1];
rz(-2.3674129495378886) q[1];
ry(-1.791750127844759) q[2];
rz(-2.300917375424824) q[2];
ry(-0.290552010052004) q[3];
rz(0.4787431031588456) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.9570036631917183) q[0];
rz(0.5170688701300155) q[0];
ry(-1.1917717845929943) q[1];
rz(1.6076627250514655) q[1];
ry(0.05054750586578157) q[2];
rz(1.5440805071084949) q[2];
ry(-1.9058369696374324) q[3];
rz(-0.17330039942744568) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.706032729752958) q[0];
rz(-1.9767412745311064) q[0];
ry(-2.3214877228957196) q[1];
rz(1.2158740919111557) q[1];
ry(-0.08250338909522821) q[2];
rz(2.324430685402944) q[2];
ry(-2.1570760619768254) q[3];
rz(-0.9533245126800703) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.860899973928702) q[0];
rz(-1.6588925572788762) q[0];
ry(-0.8585853513453983) q[1];
rz(-2.4247829064448996) q[1];
ry(-1.29231949724714) q[2];
rz(-0.9226798804589312) q[2];
ry(-0.5669359333676995) q[3];
rz(-1.9134296607024615) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.163276963931482) q[0];
rz(-0.17833447639158553) q[0];
ry(-1.9161917571626819) q[1];
rz(0.8431766695011861) q[1];
ry(-2.929072108318307) q[2];
rz(-0.2875104666849703) q[2];
ry(-1.7974346513001322) q[3];
rz(-1.4582614020573796) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.10095432550698538) q[0];
rz(0.6657274729987321) q[0];
ry(2.0074920603139836) q[1];
rz(-2.403297634380525) q[1];
ry(1.6382036265170683) q[2];
rz(2.1163351219215434) q[2];
ry(0.000336668048674185) q[3];
rz(1.034384829481227) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.5037739261116415) q[0];
rz(2.9998897234889883) q[0];
ry(-2.5885910527407328) q[1];
rz(1.4770397086618565) q[1];
ry(3.0264956649849464) q[2];
rz(-1.5096098328481207) q[2];
ry(0.5054186624965726) q[3];
rz(-2.352314109432112) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8185628600765833) q[0];
rz(2.52995974430636) q[0];
ry(-0.5935865106778025) q[1];
rz(-1.8656019180058072) q[1];
ry(-1.0683345912496165) q[2];
rz(-0.8242681615417237) q[2];
ry(1.0567063217274226) q[3];
rz(2.3669792101843172) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.20818036599931355) q[0];
rz(3.0488828935615504) q[0];
ry(-0.5681082484892981) q[1];
rz(0.19232497709369767) q[1];
ry(-2.3534765209149606) q[2];
rz(-1.0790548102600457) q[2];
ry(-0.6956536599861395) q[3];
rz(-1.6541514020541872) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.4853941720993546) q[0];
rz(0.4325659479405144) q[0];
ry(-2.3858879984840997) q[1];
rz(-1.329466027947002) q[1];
ry(-2.9271358224138724) q[2];
rz(-2.076872458292395) q[2];
ry(2.309527019134416) q[3];
rz(-1.0342959488378647) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.7802186328467506) q[0];
rz(-2.147689274826069) q[0];
ry(-1.4707352465761718) q[1];
rz(-2.5066041163252275) q[1];
ry(-0.4444703467459238) q[2];
rz(-2.86891216260856) q[2];
ry(-0.746849197980374) q[3];
rz(-0.4739691244235961) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9260308638501691) q[0];
rz(1.4952667426555861) q[0];
ry(0.40368856274814396) q[1];
rz(0.8136916735232316) q[1];
ry(0.3580386918126379) q[2];
rz(-0.3095730322120809) q[2];
ry(-2.849807440811079) q[3];
rz(0.5531644178014339) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4627555604041254) q[0];
rz(1.4246861953824315) q[0];
ry(2.8872217223244165) q[1];
rz(-1.4325012117058433) q[1];
ry(-1.5524792314723301) q[2];
rz(1.8121627595399374) q[2];
ry(2.250807050345384) q[3];
rz(0.5304867664321068) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.8746148097876023) q[0];
rz(0.452560522473064) q[0];
ry(-1.9742526188271552) q[1];
rz(-1.3417829225712579) q[1];
ry(-2.7235105825585135) q[2];
rz(0.92055485091492) q[2];
ry(2.665482693215181) q[3];
rz(-3.097167210899335) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4543240678530558) q[0];
rz(1.2960877392044963) q[0];
ry(1.344963281671337) q[1];
rz(0.6408200083020666) q[1];
ry(1.864655381275611) q[2];
rz(-2.7156926100633374) q[2];
ry(1.0348621297372176) q[3];
rz(0.44995085632861453) q[3];