OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04419413961577721) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10893594147866198) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.23732480747763088) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.41143563209390577) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.3614870446558861) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.001271123034349827) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0013605017124563661) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.1870542193762489) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.002612543549373219) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.003179689389316651) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.7409184710530877) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.5060585663295102) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.0017924634008342156) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.057948510563268656) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.3141976224069271) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.0012758307742322895) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.00017932407302663015) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.29474456581798275) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.5872226922798225) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.6610112903959993) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.5674601193691823) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.6739237226346614) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.133184286978746) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.7645624469464007) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.09289172047236506) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.3966470506241931) q[11];
cx q[10],q[11];
rx(-0.44113561229638093) q[0];
rz(-0.21873959235524154) q[0];
rx(0.006574798632498064) q[1];
rz(-0.05513177237319388) q[1];
rx(0.002255285997275141) q[2];
rz(-0.2987771159751514) q[2];
rx(-0.9979111455366849) q[3];
rz(-0.0093453608147785) q[3];
rx(-1.697222749706165) q[4];
rz(0.006397821690959212) q[4];
rx(-2.9989876327409808e-05) q[5];
rz(0.15385290918251035) q[5];
rx(-1.0926228476410158) q[6];
rz(0.007816898257375079) q[6];
rx(-0.00022666973794303633) q[7];
rz(-0.04709120619320668) q[7];
rx(-0.0008271752482432004) q[8];
rz(-0.11861330709079956) q[8];
rx(0.0016840234113684553) q[9];
rz(-0.38787477074437265) q[9];
rx(0.0007815988692219315) q[10];
rz(-0.5531629968830696) q[10];
rx(0.056052557528558006) q[11];
rz(0.09001975908825205) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2193454186241004) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0007611313702366885) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.21874938018347137) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.35010733766467095) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.008795788757122187) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-6.986061366114803e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(5.254022620482243e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.0033401812252131283) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.002533406765077786) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0004691775856852468) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.3374621410922842) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.003413069481134464) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-1.1081700282567046) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.4215705030140868) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.11998212755082988) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.19755344832551) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.32117763456723347) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-1.2120137160580378) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.07418641051898964) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.0640172643732284) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.3598555034190702) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.19899474744715087) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.11924605689234069) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.7430258708923171) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.11242655514810493) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.09352713440991531) q[11];
cx q[10],q[11];
rx(-0.5336673213382537) q[0];
rz(0.04579027947220663) q[0];
rx(0.0026010029339203653) q[1];
rz(-0.018745717829626373) q[1];
rx(0.0975715265721813) q[2];
rz(-0.43975635566312465) q[2];
rx(-1.125026167515434) q[3];
rz(0.0184771925177301) q[3];
rx(-1.265667167316311) q[4];
rz(-0.18541186215509955) q[4];
rx(-0.0006217763832461038) q[5];
rz(-0.006183580718279953) q[5];
rx(0.0008413594490220087) q[6];
rz(-0.6503369514567982) q[6];
rx(0.0011209747327024606) q[7];
rz(-0.8162671757879013) q[7];
rx(-0.00029213860526656533) q[8];
rz(-0.3003947034178941) q[8];
rx(0.0028271396621962403) q[9];
rz(-1.4299967333021282) q[9];
rx(-0.000880100694320653) q[10];
rz(-0.5505895357019844) q[10];
rx(-0.06538342599084127) q[11];
rz(-0.057359982175473306) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.7412060712375669) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.13173053067329146) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.14640616773413676) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.3369460938614674) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.007065199466535419) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.810745403642885e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-7.495957437695018e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.005489692329599756) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.010823819900618838) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-1.4288576238781907e-05) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.3253657712811353) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.01573607794897384) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.026694121665587277) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-1.5764505093419792) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.05667681885849232) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.14210266552127088) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.14284438600565533) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.004948238039671728) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.06887646636847385) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.05900019940881604) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.7830875029237584) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.1365501897436909) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.04653219968148878) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.3325497040634957) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(0.38997900622929166) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.037592082695470444) q[11];
cx q[10],q[11];
rx(-0.9259068726231818) q[0];
rz(0.007477980065253989) q[0];
rx(-0.01064161469509776) q[1];
rz(-0.35500653898474926) q[1];
rx(-0.10895286625224933) q[2];
rz(-0.5168057017320019) q[2];
rx(-1.0196924562001695) q[3];
rz(0.43716999300494686) q[3];
rx(0.002230103896216816) q[4];
rz(-0.4176553202599157) q[4];
rx(0.0005667185787169105) q[5];
rz(-1.3895968750507997) q[5];
rx(0.0007640054672916768) q[6];
rz(-0.27601238100830816) q[6];
rx(-0.001799838370713613) q[7];
rz(-0.1502483078857037) q[7];
rx(-0.6719833198396279) q[8];
rz(0.00018007709129273805) q[8];
rx(-0.0007866080023603516) q[9];
rz(-0.23694663255181458) q[9];
rx(0.00016421350145049045) q[10];
rz(-0.41381494383695494) q[10];
rx(-1.277717348863345) q[11];
rz(-0.000257631861513654) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.35279729482046507) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08518456346902256) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.030719563045604983) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04275271579080628) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0901877403958127) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.022701826829973886) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.023689127011087843) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.20468453982079737) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.1264866321045425) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0981257978808714) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.4083090381459597) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.22943854656999793) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.23771776706401257) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.5138996698144902) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.8177117091686232) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(0.17441655017885366) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.0006994554749095046) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.015986755911248648) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.22867882548462182) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-9.563004513086967e-05) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.5375591468499727) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.0021141847797898606) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.10505538779558796) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.01473965461264833) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.001383120145574308) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.08557494190800712) q[11];
cx q[10],q[11];
rx(-0.7901404821980433) q[0];
rz(0.10463169372444228) q[0];
rx(0.007024170409636427) q[1];
rz(-0.36817008653074984) q[1];
rx(-0.00825808827451431) q[2];
rz(-0.4838854603405478) q[2];
rx(-0.0007209505106166356) q[3];
rz(-0.18276225086070103) q[3];
rx(0.0016704048733446782) q[4];
rz(-0.2236375596899648) q[4];
rx(-0.00022471695438529456) q[5];
rz(-0.15170915428972054) q[5];
rx(3.3551953283479964e-05) q[6];
rz(-0.37840576047750807) q[6];
rx(-1.3841752491669745) q[7];
rz(0.00023240522549195522) q[7];
rx(-1.5306967316711266) q[8];
rz(-4.4448330802205105e-05) q[8];
rx(-0.0005270876907080886) q[9];
rz(-0.0020615485848350795) q[9];
rx(-0.0016667942860850713) q[10];
rz(-0.2250154183825168) q[10];
rx(-1.6040626437221979) q[11];
rz(0.021062253380610522) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.33793719426213137) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1247742375531685) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.19519327684949786) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.4358955607157152) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.4313849207709803) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.9525457585928694) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.949919725155952) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.01302446502941915) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7100798981078774) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.6720379456166538) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.45579652388077446) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.4633163716925436) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.06816898906231005) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.03691222480301733) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.00025210401583173) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.000119931242422029) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(7.903265268963939e-06) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.00021437460987399452) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.17323996118982132) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(1.3744165539044531e-05) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.009890241080651108) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.0004716998626250496) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.07758293543975599) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.05249825541080528) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.041453491914586665) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.3209855050071021) q[11];
cx q[10],q[11];
rx(-0.5079235140228466) q[0];
rz(0.16261421350479996) q[0];
rx(-0.00348253573746521) q[1];
rz(-0.27220979643835597) q[1];
rx(8.311936747785761e-07) q[2];
rz(-0.5317225230984437) q[2];
rx(-0.00012830414785968273) q[3];
rz(-0.007979599574686856) q[3];
rx(-0.00011534826838488842) q[4];
rz(-0.4382519101800381) q[4];
rx(0.00019361020968139718) q[5];
rz(0.04851008698653445) q[5];
rx(-0.0002380428441134076) q[6];
rz(-0.3918464519726839) q[6];
rx(-1.756640205321119) q[7];
rz(0.12706179998738104) q[7];
rx(-0.962614901475455) q[8];
rz(0.10658275517307798) q[8];
rx(-0.0002389290858023206) q[9];
rz(-0.05471780061330444) q[9];
rx(0.001592981314820423) q[10];
rz(-0.4904421100378617) q[10];
rx(-0.26665762956407046) q[11];
rz(-0.15881441376179514) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.5715211978271223) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.5796085814017116) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.017233709301408887) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.961713205832423) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.9346534991949024) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.2520549426097447) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.24758602530792934) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.14325295600918742) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.30669011317166267) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.21086504700677688) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.6503476307585728) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.6547392209456235) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.2712869831129708) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.003293658277726544) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.07143522487311618) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.23891433234900805) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2462024530659395) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.04605727980082594) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.15242830004933083) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.665216952626578) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.11786267423411802) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(0.07939056674513581) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.07660034903201773) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.289729221039824) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-1.159230649807248) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.12313278081633162) q[11];
cx q[10],q[11];
rx(0.020893689130736706) q[0];
rz(-0.049785894028133264) q[0];
rx(0.0006591702502096933) q[1];
rz(0.06259611440041474) q[1];
rx(0.0003138325507675818) q[2];
rz(-0.0317823404197251) q[2];
rx(0.00021460905375866803) q[3];
rz(0.04000781371898526) q[3];
rx(-2.916031245828001e-05) q[4];
rz(-0.05826386342653374) q[4];
rx(-4.035820105905308e-05) q[5];
rz(0.06521970059494835) q[5];
rx(0.0007275381570005424) q[6];
rz(-0.06867162049150688) q[6];
rx(9.777576427080076e-05) q[7];
rz(-0.07615811415811896) q[7];
rx(-0.00034681012254312286) q[8];
rz(-0.21915083913268057) q[8];
rx(-0.0001480298893503478) q[9];
rz(-0.032955591204886274) q[9];
rx(0.0005144318397452664) q[10];
rz(-0.6591584310306212) q[10];
rx(0.0005417016015761035) q[11];
rz(0.09373529470633302) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-1.0104959063622463) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.9905898834074953) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.18129779688343672) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.545091003794596) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.5506961840206646) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.1117693498756434) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.10157356838260084) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.08150741300144752) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.3134157091109935) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.3421981022000428) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.21123078855593724) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.19732690555659735) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.13015015523949616) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.18551595735204454) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.15245390879915827) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.13345187282922755) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.13489572358414403) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.06208832714690719) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.5955611552957242) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.7252269009807055) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.2102537095051884) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-1.268682097783966) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.023026515327057892) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.20795714163504597) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.2574354736021103) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.845659770398342) q[11];
cx q[10],q[11];
rx(0.0009143476673463318) q[0];
rz(-0.027215703601001773) q[0];
rx(-0.0017539808452813944) q[1];
rz(0.06861819842260418) q[1];
rx(-0.00014620759970291073) q[2];
rz(-0.04327657462126252) q[2];
rx(-0.0001504583319182561) q[3];
rz(0.09134703682781722) q[3];
rx(0.00010844465175370562) q[4];
rz(-0.110647206560899) q[4];
rx(-0.0001619965340658976) q[5];
rz(0.07900752435374085) q[5];
rx(-0.0003157665223092626) q[6];
rz(-0.04955721197152335) q[6];
rx(0.00020053330717105576) q[7];
rz(0.010496690680589643) q[7];
rx(-0.00010214285068378993) q[8];
rz(0.1887043668889205) q[8];
rx(-0.0002384533127817966) q[9];
rz(0.018365883157043912) q[9];
rx(-0.0020337098151376213) q[10];
rz(-0.00587901801613431) q[10];
rx(0.00041584160902408736) q[11];
rz(-0.04929798494333687) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.24368141471152718) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.24711375975358987) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.12656032215331697) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.054286197487347684) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.008648038471587308) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5609608941948048) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.5574175687210966) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.0046033266732992155) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.685496930108504) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.6508140619247199) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.04656499971631656) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.06316448419013002) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.005576591324993336) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.13589214368935765) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.12831124785337664) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
h q[6];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.2532349142323868) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
h q[8];
sdg q[6];
h q[6];
sdg q[8];
h q[8];
cx q[6],q[7];
cx q[7],q[8];
rz(-0.21986223878962974) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.058870328442474876) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.5690331651883334) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
h q[9];
sdg q[7];
h q[7];
sdg q[9];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.7901313246694798) q[9];
cx q[8],q[9];
cx q[7],q[8];
h q[7];
s q[7];
h q[9];
s q[9];
h q[8];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.0775056163397822) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
h q[10];
sdg q[8];
h q[8];
sdg q[10];
h q[10];
cx q[8],q[9];
cx q[9],q[10];
rz(-0.07431949210045405) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.0013638371337538556) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.11117327886673017) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
h q[11];
sdg q[9];
h q[9];
sdg q[11];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.12008891340819172) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.7432909352571561) q[11];
cx q[10],q[11];
rx(-0.00428098326981403) q[0];
rz(-0.05453230502102228) q[0];
rx(0.002051037400317987) q[1];
rz(-0.0038765138809590824) q[1];
rx(0.0002445394745735782) q[2];
rz(-0.023504443129848826) q[2];
rx(-0.000251200100790223) q[3];
rz(-0.03580277444872837) q[3];
rx(-4.1565136888766584e-05) q[4];
rz(-0.053709710527997556) q[4];
rx(0.0003956045233720747) q[5];
rz(-0.07690282044018536) q[5];
rx(0.00011409190020541068) q[6];
rz(-0.05236909230128638) q[6];
rx(-0.0005779266995083906) q[7];
rz(-0.024123701206647048) q[7];
rx(0.0001361466328901955) q[8];
rz(-0.05555440938767429) q[8];
rx(0.0005144731079250336) q[9];
rz(-0.07490250148550784) q[9];
rx(0.0011174777842911402) q[10];
rz(-0.0066625902430040695) q[10];
rx(0.00441121750085621) q[11];
rz(-0.12121104463724178) q[11];