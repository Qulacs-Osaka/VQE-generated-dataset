OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.050219884859503836) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.01101871075605175) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02634518705536558) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.008562211400064032) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.038385286788741375) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.025029931904632752) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06017190488760371) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0002442143715274309) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09771641895062651) q[3];
cx q[2],q[3];
rz(-0.055486910246064045) q[0];
rz(-0.07034681776668604) q[1];
rz(-0.06075025884512734) q[2];
rz(-0.04212467452066165) q[3];
rx(-0.1921002354015747) q[0];
rx(-0.036502226765477506) q[1];
rx(-0.09148496426214951) q[2];
rx(-0.0352579647877537) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.027066110106691604) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09407613996699662) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04675738409838015) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.023968958227043802) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11598984609153125) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0255209494762522) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.01913377068816676) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0030288180030629713) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.020378036239810604) q[3];
cx q[2],q[3];
rz(-0.04594895927491863) q[0];
rz(-0.08067285044227351) q[1];
rz(-0.014850213793912754) q[2];
rz(-0.0203208277891005) q[3];
rx(-0.10754360960909465) q[0];
rx(-0.1342396339277529) q[1];
rx(-0.024390711787462212) q[2];
rx(0.003652874117293753) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05006181241069243) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07154265450445227) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.026537384555922163) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04558497248820287) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06614738877999837) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.00010290560347797535) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05470681431353397) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.02070223695386985) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.00013176716884064288) q[3];
cx q[2],q[3];
rz(-0.045616923013334865) q[0];
rz(-0.051337510603298664) q[1];
rz(-5.5671697488489406e-05) q[2];
rz(0.042727140351535145) q[3];
rx(-0.1508778159310674) q[0];
rx(-0.14103355795541203) q[1];
rx(-0.020900255965154386) q[2];
rx(-0.006190861685939859) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.00899586387513877) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06798972900526516) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04955066062132813) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08684540653126109) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13959434850759841) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0730932874853383) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0008339250969862106) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03562720791067953) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05773486247994546) q[3];
cx q[2],q[3];
rz(0.05242900944760693) q[0];
rz(0.007673665193867437) q[1];
rz(-0.030056971282347432) q[2];
rz(0.0618225634313896) q[3];
rx(-0.11977413641809674) q[0];
rx(-0.19317889014939316) q[1];
rx(-0.018015292245491997) q[2];
rx(-0.02004522680504892) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0015520868123341063) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1700808956304125) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.017469261201562374) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08243651731699961) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.20095892226077394) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.008528736556514885) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.062490037314982254) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0012516470135584762) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07595132884309372) q[3];
cx q[2],q[3];
rz(0.04719464405058561) q[0];
rz(0.03167360983180355) q[1];
rz(-0.07586572854789345) q[2];
rz(0.09574378527969561) q[3];
rx(-0.15226220392749026) q[0];
rx(-0.14606481213884806) q[1];
rx(-0.058988098396084925) q[2];
rx(0.002382727143548254) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07960267662340656) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.20103208499549757) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.025091342991961283) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09013143290937786) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.1893339162518233) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.000734335495188072) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.04628709660170554) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.049534964346963084) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12175442242224035) q[3];
cx q[2],q[3];
rz(0.05008929211886476) q[0];
rz(0.024125471729327073) q[1];
rz(-1.6100049103491494e-05) q[2];
rz(0.10278728457513367) q[3];
rx(-0.15036750001353216) q[0];
rx(-0.22248603340928907) q[1];
rx(-0.09088825679099004) q[2];
rx(-0.05886607325866624) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.051495979684776004) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1568773382002935) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02699338464113142) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.12082492408901246) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.21727797678270297) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.069777786932228) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.12210985348450551) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.07478885238189285) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1372020082144691) q[3];
cx q[2],q[3];
rz(0.09841574029685338) q[0];
rz(0.02743020261628438) q[1];
rz(0.004859216999499705) q[2];
rz(0.12552924156626805) q[3];
rx(-0.14026060536586796) q[0];
rx(-0.2220638856153522) q[1];
rx(-0.07859699195527688) q[2];
rx(-0.06823926775724844) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05493438486948086) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.20753923093304227) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028832327922065717) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08720797064079047) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19000788903659688) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06635193950032801) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.11596038115727524) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03272211005030017) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03495559102792058) q[3];
cx q[2],q[3];
rz(0.053622516698848655) q[0];
rz(0.029134564975565706) q[1];
rz(-0.03783870276144207) q[2];
rz(0.05875416756647327) q[3];
rx(-0.04908382803862194) q[0];
rx(-0.17197641898867402) q[1];
rx(-0.12698495767163198) q[2];
rx(-0.03032193959062893) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03213596806565158) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23594635164186017) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.026060244559231182) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10023006425407276) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19382278881700402) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.04299228718296681) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.050950450209034784) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.09777394412928729) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03336416935106287) q[3];
cx q[2],q[3];
rz(0.025605349826383052) q[0];
rz(-0.02487048763797702) q[1];
rz(0.007557154604957088) q[2];
rz(0.08346209006070911) q[3];
rx(-0.056897614233049586) q[0];
rx(-0.18607354288565892) q[1];
rx(-0.17680524805592285) q[2];
rx(-0.05144424736374387) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03873581003442438) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2549669186840207) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06164893721086813) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04585112538557918) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11189053558547778) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08860951028617703) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.045287299923509425) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.006403248961783703) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02368615004727048) q[3];
cx q[2],q[3];
rz(0.03155804688489358) q[0];
rz(0.020526571170021515) q[1];
rz(-0.002463507687668959) q[2];
rz(0.12174520183471173) q[3];
rx(-0.043646960819030016) q[0];
rx(-0.14410930765947616) q[1];
rx(-0.1607751371967707) q[2];
rx(-0.12713634688642408) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07138209466112247) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.20163909606818403) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.009897103593106098) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04842019056286739) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07857535620331825) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.11373184363299924) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06657109414904179) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0427662180344857) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0794369287110614) q[3];
cx q[2],q[3];
rz(0.07677453659171653) q[0];
rz(-0.0021244003856686518) q[1];
rz(0.028367764264155465) q[2];
rz(0.07924826382238889) q[3];
rx(-0.02233175948460236) q[0];
rx(-0.22802864300623485) q[1];
rx(-0.19035445570435156) q[2];
rx(-0.09646861651399934) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.04751817240706822) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1036254678311314) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.006866049303275204) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04510328511984993) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03754334195999746) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13531528087381733) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0333492274424675) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0493312906816206) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07115161464789853) q[3];
cx q[2],q[3];
rz(0.07826701114038075) q[0];
rz(0.06819625409337923) q[1];
rz(0.012331578407412414) q[2];
rz(0.07471959108311554) q[3];
rx(-0.02196178256372617) q[0];
rx(-0.22662253839007374) q[1];
rx(-0.12504803369433362) q[2];
rx(-0.1585937858860617) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08530465247526664) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10506758360303738) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.022501756245980095) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.017348177489742687) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03496960514753716) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08067181418041838) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.017870721213777196) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.024389062818310978) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08954276975478799) q[3];
cx q[2],q[3];
rz(0.07863556708478163) q[0];
rz(0.057997107712054144) q[1];
rz(-0.002901338991500851) q[2];
rz(0.09897204989377913) q[3];
rx(-0.030531474245457074) q[0];
rx(-0.1921538093522614) q[1];
rx(-0.19140166842851386) q[2];
rx(-0.19239406120502767) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1642307994894589) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0527071570283023) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-8.787142587955405e-05) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.05768380689073054) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.031057809860075357) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.12736402431305108) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.005280726469395991) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.03023807292323432) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06202292238626588) q[3];
cx q[2],q[3];
rz(0.0739891760412223) q[0];
rz(-0.006931112590351665) q[1];
rz(-0.0890310957936931) q[2];
rz(0.043796088796800414) q[3];
rx(-0.09344563493220244) q[0];
rx(-0.20486088121870932) q[1];
rx(-0.16820786138273838) q[2];
rx(-0.1797017885850519) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10888590820265699) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.04629908706182141) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.014850055825743442) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.08063841053246229) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.08446920082922524) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.14531341522511101) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04799587182816352) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10803179313747002) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11016396194530585) q[3];
cx q[2],q[3];
rz(0.027629525974661995) q[0];
rz(-0.06755790923186034) q[1];
rz(-0.09255899901382428) q[2];
rz(0.05358628194094309) q[3];
rx(-0.11599241211312912) q[0];
rx(-0.27880043836297375) q[1];
rx(-0.13831373880199668) q[2];
rx(-0.18969127881614922) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10075774516653299) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.15735286315618327) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06203785807721712) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11965434624718067) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1006441157419803) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0910484001209764) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06941342522813604) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.04248420516326176) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08672727662884527) q[3];
cx q[2],q[3];
rz(-0.0070559342397130105) q[0];
rz(-0.13298296896057907) q[1];
rz(-0.1620312554826694) q[2];
rz(0.061643405694662784) q[3];
rx(-0.05713453450695727) q[0];
rx(-0.24735490286906148) q[1];
rx(-0.1704691253994407) q[2];
rx(-0.22577471327803972) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06087369287585148) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08742297674363506) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03990205252904141) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1583818261127287) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.042454555471066684) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.059917101905123546) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10905198506959267) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.015152566814020748) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05152125033886124) q[3];
cx q[2],q[3];
rz(-0.10142316446760664) q[0];
rz(-0.15540644313072854) q[1];
rz(-0.20479784998041242) q[2];
rz(0.019990482812166754) q[3];
rx(-0.08361523532686989) q[0];
rx(-0.1485980111576945) q[1];
rx(-0.1251940522312956) q[2];
rx(-0.15031630637972965) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03661127474235114) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0987555741065426) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0030602620881014305) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12745781700515477) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.02673564223096111) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0627993125729649) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.13903085012729618) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.009886553341709077) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0047333896442096615) q[3];
cx q[2],q[3];
rz(-0.06864472962735284) q[0];
rz(-0.13849802802770464) q[1];
rz(-0.17419698717595436) q[2];
rz(-0.01201388628724885) q[3];
rx(-0.08876933290806563) q[0];
rx(-0.19768947120874603) q[1];
rx(-0.14183760816567942) q[2];
rx(-0.21732351185935134) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.060472062385059505) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08458984454469637) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.012086060241498586) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15389089823158503) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07841462260811793) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08780772120553157) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1565930079527414) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0813076230519432) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.013383834657202278) q[3];
cx q[2],q[3];
rz(-0.1033856376203418) q[0];
rz(-0.1596113509564032) q[1];
rz(-0.17949639775116433) q[2];
rz(-0.030451279382358264) q[3];
rx(-0.17231723855126993) q[0];
rx(-0.15200868383838165) q[1];
rx(-0.13123932120497012) q[2];
rx(-0.24054317798896077) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0032679901177166456) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0005594826622781788) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.021146756011376996) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1542204450296511) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.11985251699581231) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05357894775815689) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.17411664131855184) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0397433865943955) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06072151266779454) q[3];
cx q[2],q[3];
rz(-0.17398970297301244) q[0];
rz(-0.08950393116359187) q[1];
rz(-0.13513393009566219) q[2];
rz(-0.02595640493027309) q[3];
rx(-0.21669627010825096) q[0];
rx(-0.1062596629746193) q[1];
rx(-0.061527751110028436) q[2];
rx(-0.22715893857471214) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.059291627982821886) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.01659487961236123) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03837470590312237) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1123088959358681) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.171857107547919) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.010477935018100308) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.23898292491572448) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0831309175092544) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1140474372453119) q[3];
cx q[2],q[3];
rz(-0.2699297461490213) q[0];
rz(-0.08489001065236249) q[1];
rz(-0.07531663842539377) q[2];
rz(-0.08801458396521777) q[3];
rx(-0.25189182014778094) q[0];
rx(-0.07554748855693538) q[1];
rx(-0.11298623455897662) q[2];
rx(-0.2413008406468617) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12498894654615936) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.028758931385626657) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06981964252062874) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15926219443596726) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19007365289471928) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.019611017662833373) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2195666336520734) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.026562227716183964) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10358960678336868) q[3];
cx q[2],q[3];
rz(-0.2843170455548571) q[0];
rz(-0.018752127238632236) q[1];
rz(0.02614918846490871) q[2];
rz(-0.06862539317743571) q[3];
rx(-0.2697352100156053) q[0];
rx(-0.12469190918949737) q[1];
rx(-0.07171277121120935) q[2];
rx(-0.2517665408662388) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13723819349275782) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.021254303892086816) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06305640439926294) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2363929513246126) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.12250029832995848) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08394064228888888) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.25104156228060803) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0019489847466760588) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08320113059778285) q[3];
cx q[2],q[3];
rz(-0.31996091756570144) q[0];
rz(-0.026176458457382812) q[1];
rz(0.11095565567274734) q[2];
rz(-0.1141505499070465) q[3];
rx(-0.18064583645667726) q[0];
rx(-0.09929562449971614) q[1];
rx(-0.11298441988306067) q[2];
rx(-0.29324770157003155) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09205709140493872) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.029696985587082045) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03478354282511933) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18889811692902614) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09884091465055307) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.025805104160051908) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.23407740181542083) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0270172451587663) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03435591805988629) q[3];
cx q[2],q[3];
rz(-0.3128889370157627) q[0];
rz(0.03040643275764352) q[1];
rz(0.23292024200970526) q[2];
rz(-0.15274003856310495) q[3];
rx(-0.2331625326824121) q[0];
rx(-0.08934340814658122) q[1];
rx(-0.0998387059597825) q[2];
rx(-0.3176811028477145) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.16429277037196965) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07379549964662939) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.00970465926044488) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22555822598754344) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.025751468122271526) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0625409031998813) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.25947121274945023) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.08924655582087786) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.049591445027440104) q[3];
cx q[2],q[3];
rz(-0.369566481261338) q[0];
rz(0.004509910124724807) q[1];
rz(0.14601035911077018) q[2];
rz(-0.1317393122481243) q[3];
rx(-0.22471538801716825) q[0];
rx(-0.11933537949199073) q[1];
rx(-0.07507200877662906) q[2];
rx(-0.3289762274150162) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1200120975354523) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.03873545457275396) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08334921277785247) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.21729375098381568) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.06764385366501727) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0406324122022933) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.24645618940322706) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0963740518707514) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05576220561307022) q[3];
cx q[2],q[3];
rz(-0.412533815356428) q[0];
rz(0.023980892116060394) q[1];
rz(0.15083310276556064) q[2];
rz(-0.15696668168252842) q[3];
rx(-0.15143407489009894) q[0];
rx(-0.013020492159334788) q[1];
rx(-0.06502086332489741) q[2];
rx(-0.3236770072597661) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12935432135398764) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04338900530713902) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11337405097385043) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.19612030180067128) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1420695324976283) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.033953856841798934) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.31421363774871774) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.086200058180681) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08661766232069876) q[3];
cx q[2],q[3];
rz(-0.31407970918685457) q[0];
rz(-0.0026588345465665708) q[1];
rz(0.05855383290815072) q[2];
rz(-0.08277558475693357) q[3];
rx(-0.1712450646749002) q[0];
rx(-0.00025223561243329826) q[1];
rx(-0.054263422437288336) q[2];
rx(-0.31039295695593616) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.19284908400066403) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.042093455845109345) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0033733501018908766) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.26182973598378523) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.13470766207267287) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19758886615842636) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.25838428809427944) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0447610154485642) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1087090197838345) q[3];
cx q[2],q[3];
rz(-0.28553708455001275) q[0];
rz(-0.09802268832815145) q[1];
rz(-0.06019340609862744) q[2];
rz(-0.06575574216727613) q[3];
rx(-0.07143454962252513) q[0];
rx(-0.015285153114019344) q[1];
rx(0.002011769611397819) q[2];
rx(-0.31014469393700966) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.22530369001665354) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.049570187947652035) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04273578341334879) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23563770608089135) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.06049247305659762) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.20509930802959014) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.218226098297251) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.055416082464714224) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0246202641299521) q[3];
cx q[2],q[3];
rz(-0.2319415414223117) q[0];
rz(-0.0982043925715881) q[1];
rz(-0.21579395929966227) q[2];
rz(-0.12019046394059034) q[3];
rx(-0.11359166559101613) q[0];
rx(0.0030748823585560496) q[1];
rx(-0.019715836901172713) q[2];
rx(-0.3218411327269942) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.23229823397045127) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03564168857198768) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07527465280171342) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14809549471232902) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.14614633911606426) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2893719461308358) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14182478360072756) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15661618445694983) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04043550794693088) q[3];
cx q[2],q[3];
rz(-0.1598793731881775) q[0];
rz(-0.10804329913262127) q[1];
rz(-0.2627123522632122) q[2];
rz(-0.02263953920454094) q[3];
rx(-0.06802072114921782) q[0];
rx(-0.07513821564283037) q[1];
rx(-0.053758558583478847) q[2];
rx(-0.2687713547346055) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.244287161425981) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.006024556511332341) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01304472306645917) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.163952177088547) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11560766903791864) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.21948079621806096) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.21362952775162738) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11291490575616239) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07042695496326339) q[3];
cx q[2],q[3];
rz(-0.09605895801315317) q[0];
rz(-0.1528391657992881) q[1];
rz(-0.2503288732191704) q[2];
rz(0.06268778083468467) q[3];
rx(-0.06980775689035448) q[0];
rx(-0.10666707292038045) q[1];
rx(-0.04426295023276894) q[2];
rx(-0.25091556857518765) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2889740027566322) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.029010431676161623) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09140977069968903) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1899567464456431) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24187609780348696) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.24755970775896033) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2037279009566196) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.028744602841765433) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.025221392240467103) q[3];
cx q[2],q[3];
rz(-0.04950485323354639) q[0];
rz(-0.10391501343688603) q[1];
rz(-0.15734839718234342) q[2];
rz(0.01547003665694347) q[3];
rx(-0.1422411922339526) q[0];
rx(-0.04388819939265995) q[1];
rx(-0.052057441515506675) q[2];
rx(-0.2945674250452434) q[3];