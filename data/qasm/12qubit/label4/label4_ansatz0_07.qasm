OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06391152936740893) q[2];
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
rz(-0.004392105853444221) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.2495174290979824) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13392584642118788) q[3];
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
rz(0.011188551585047984) q[3];
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
rz(0.08894473504127283) q[4];
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
rz(0.09467802280888875) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.015994545159864423) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.045790031835128656) q[5];
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
rz(-0.05852494633803552) q[5];
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
rz(-0.4538504286654898) q[6];
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
rz(0.4864950985121689) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.11644313188840845) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2602016551013368) q[7];
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
rz(0.011966374274659443) q[7];
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
rz(0.00912130218801644) q[8];
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
rz(-0.008572776879979429) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.6228013828720167) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.0003870155653762822) q[9];
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
rz(-0.003184627753511603) q[9];
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
rz(-0.251400758993983) q[10];
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
rz(0.15162646878290817) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.8700274871126527) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.19413623580266845) q[11];
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
rz(0.12217937606279852) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.19047897006189504) q[11];
cx q[10],q[11];
rx(-0.4583834808441272) q[0];
rz(0.08080814422113133) q[0];
rx(-0.007514449328122695) q[1];
rz(-0.0318921207374974) q[1];
rx(-0.03985920204712005) q[2];
rz(0.10162200743948605) q[2];
rx(-0.2754755304629156) q[3];
rz(-0.04066894479074709) q[3];
rx(0.0213498972548765) q[4];
rz(0.06436887294668484) q[4];
rx(-0.0006423540880127009) q[5];
rz(-0.39750317024178516) q[5];
rx(-0.40284173988754646) q[6];
rz(-0.08865709728567392) q[6];
rx(-0.04296544689539873) q[7];
rz(-0.16298829225590808) q[7];
rx(-0.6673136443974718) q[8];
rz(-0.3658693609300606) q[8];
rx(-0.021756766139882238) q[9];
rz(-0.4378182553433597) q[9];
rx(-0.04123776543274589) q[10];
rz(-0.1352895671929541) q[10];
rx(-0.41659412864878725) q[11];
rz(-0.1299399215126209) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.4807421914867535) q[2];
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
rz(0.09261080347407172) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.015120666406981271) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12875442720507288) q[3];
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
rz(-0.0270362669616866) q[3];
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
rz(0.007694448839302001) q[4];
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
rz(0.002776854052437516) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.0007983836563738548) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.045848608492774126) q[5];
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
rz(-0.0361849535934164) q[5];
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
rz(-0.43198763374556476) q[6];
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
rz(0.4791423579892496) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.15521411815491215) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.017772192211658173) q[7];
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
rz(-0.008362196517803842) q[7];
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
rz(-0.0020817686657869875) q[8];
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
rz(0.0016467445090597073) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.11091412715870679) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.0001298737331583402) q[9];
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
rz(-0.0004916877845422536) q[9];
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
rz(0.23012408624924627) q[10];
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
rz(-0.24337674425203445) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.11811927964761759) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.5140656185304978) q[11];
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
rz(0.18780543080632478) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.10540353653349561) q[11];
cx q[10],q[11];
rx(-0.7738948414570801) q[0];
rz(-0.06370019346007526) q[0];
rx(0.010783805609917927) q[1];
rz(0.11255826519469901) q[1];
rx(0.05558833771228787) q[2];
rz(0.04930812116787468) q[2];
rx(-0.8908088816855232) q[3];
rz(0.03628292896006712) q[3];
rx(-0.007736368815031715) q[4];
rz(-0.02554392998966721) q[4];
rx(-0.0003794004629642655) q[5];
rz(-0.017839603883164593) q[5];
rx(-1.0448466461524217) q[6];
rz(-0.06793200689823434) q[6];
rx(-0.36684870969529243) q[7];
rz(-0.038826550645298545) q[7];
rx(-0.897266703572693) q[8];
rz(0.24613526374961645) q[8];
rx(-0.051482468218646917) q[9];
rz(-0.19258939181174425) q[9];
rx(0.031890596088828056) q[10];
rz(-0.005796677097890312) q[10];
rx(-0.5401569030748562) q[11];
rz(0.059532361658129904) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.7253405948385244) q[2];
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
rz(-0.00865878907210981) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.01631611129276746) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1849582707513684) q[3];
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
rz(0.02487630255671) q[3];
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
rz(-0.0014798457645363945) q[4];
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
rz(0.0002780183086626035) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.023587289364016124) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.009783603688410459) q[5];
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
rz(0.009940099924076445) q[5];
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
rz(-0.04948572249706553) q[6];
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
rz(-0.24054148345654086) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.661199655084653) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.01353865942360069) q[7];
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
rz(8.33413474935008e-06) q[7];
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
rz(-0.0004339200244122339) q[8];
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
rz(0.001170558830965806) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.004619966608859139) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.005571950307166527) q[9];
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
rz(0.00025938430182534535) q[9];
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
rz(-0.7766582138628508) q[10];
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
rz(0.13619251428257226) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.03497465959840152) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.3217603783605396) q[11];
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
rz(0.060441118435339974) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.002303107879882451) q[11];
cx q[10],q[11];
rx(-0.7253699472264532) q[0];
rz(0.02106420065096714) q[0];
rx(-0.019952655097015676) q[1];
rz(0.010772982682228649) q[1];
rx(-0.0040398810197472945) q[2];
rz(-0.3680245712546124) q[2];
rx(-1.022264884449115) q[3];
rz(-0.02236761083660682) q[3];
rx(-0.012810922311301782) q[4];
rz(0.27541332903220467) q[4];
rx(0.00015885727971399506) q[5];
rz(-0.09072482327061102) q[5];
rx(-1.0585556645988066) q[6];
rz(0.1569839642212173) q[6];
rx(-0.8947686768409088) q[7];
rz(0.014668754177890713) q[7];
rx(-0.9466116935770307) q[8];
rz(0.0350891171079237) q[8];
rx(0.012080255553888773) q[9];
rz(-0.0691169826056701) q[9];
rx(-0.11413299248563968) q[10];
rz(-0.07084784001527372) q[10];
rx(-0.7832646492151877) q[11];
rz(0.040663302534236195) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.27963275362450596) q[2];
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
rz(0.013445337703552378) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.003357783605265038) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.25327538056624377) q[3];
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
rz(-0.020196976660688294) q[3];
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
rz(-0.0005215145304595723) q[4];
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
rz(-0.0009131014564539145) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.028166667588979215) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.014569807387237621) q[5];
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
rz(-0.002118725983221894) q[5];
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
rz(0.028543614237708986) q[6];
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
rz(0.093424472856348) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.8101023098696544) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.022273934720704644) q[7];
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
rz(0.0002476102916587731) q[7];
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
rz(0.00021435494874368001) q[8];
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
rz(0.00016358545164896083) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.0009602007682141416) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.008329075502545283) q[9];
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
rz(0.0007739342071953047) q[9];
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
rz(-1.1532569033183537) q[10];
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
rz(0.020247768524398007) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.09981139570371977) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.033078900204649736) q[11];
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
rz(-0.10819804672831476) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.023627943111012024) q[11];
cx q[10],q[11];
rx(-0.5966104766427182) q[0];
rz(0.03652080698680363) q[0];
rx(0.02316302434940065) q[1];
rz(-0.25947231283973726) q[1];
rx(-0.006650716014989486) q[2];
rz(-0.5055798540010862) q[2];
rx(-0.9553649084021867) q[3];
rz(-0.05753121036947862) q[3];
rx(0.01966367257626899) q[4];
rz(0.19761974691343126) q[4];
rx(0.0013799932948295023) q[5];
rz(0.0879647159267581) q[5];
rx(-0.7656319904891684) q[6];
rz(0.2488145085468327) q[6];
rx(-0.9438067679304475) q[7];
rz(0.011304343091471058) q[7];
rx(-0.7360353190912753) q[8];
rz(0.1692394629063602) q[8];
rx(-0.013972076620294323) q[9];
rz(-0.5081645376251768) q[9];
rx(0.10936973142292815) q[10];
rz(-0.23229035174097087) q[10];
rx(-0.9723130030576507) q[11];
rz(-0.17814563625467464) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.05756765318006732) q[2];
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
rz(-0.10769551026098871) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.014306215177610314) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16229391459119588) q[3];
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
rz(-0.3353617566659242) q[3];
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
rz(-0.09875354402901196) q[4];
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
rz(-0.0954121270463017) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.06037064473160797) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.13742110070593636) q[5];
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
rz(-0.13252813366638888) q[5];
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
rz(0.08322107296775366) q[6];
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
rz(0.009307483245681814) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.4306250488121066) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.01646291276452404) q[7];
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
rz(0.0019742216503270417) q[7];
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
rz(5.0223122898181686e-05) q[8];
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
rz(6.463794290589796e-05) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.0057461031814188554) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.005815200778484603) q[9];
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
rz(-0.0008658921414564186) q[9];
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
rz(-0.8530747206081383) q[10];
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
rz(0.26468505084903704) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.5017633579011811) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.4810427548083136) q[11];
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
rz(-0.11426409110831695) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.08330467087997023) q[11];
cx q[10],q[11];
rx(-0.48159387695843625) q[0];
rz(-0.10572167733072024) q[0];
rx(-0.00359330558115832) q[1];
rz(-0.4918549120394212) q[1];
rx(0.002558938641554433) q[2];
rz(-0.8449056284002486) q[2];
rx(0.00196820305194252) q[3];
rz(0.6034580450605749) q[3];
rx(-0.011438226034327727) q[4];
rz(0.25204862719854243) q[4];
rx(-5.772032617968302e-06) q[5];
rz(-0.6405374697579878) q[5];
rx(0.09374308392799617) q[6];
rz(0.40733501728650257) q[6];
rx(-0.8951014193877762) q[7];
rz(0.2818768999796569) q[7];
rx(0.04558689045719559) q[8];
rz(0.05658692115193893) q[8];
rx(-0.016371897870026582) q[9];
rz(-0.6077929660057301) q[9];
rx(-0.04666246810745448) q[10];
rz(-0.40696252696694585) q[10];
rx(-0.45555307373062875) q[11];
rz(0.016118919053435654) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.14396349737960323) q[2];
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
rz(0.15468405189572215) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.08424066334023914) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.3216397848537585) q[3];
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
rz(0.28906809291440705) q[3];
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
rz(-0.06795591567789983) q[4];
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
rz(-0.07317747350371909) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.011507019209174876) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.011715078861929563) q[5];
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
rz(-0.020880743544754037) q[5];
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
rz(-0.2213288203107729) q[6];
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
rz(-0.250535506761318) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.1621970620013747) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.7848072588498328) q[7];
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
rz(-0.8188471136249255) q[7];
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
rz(0.3201814600177491) q[8];
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
rz(0.3096704426600275) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.008154997316937092) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.37317230136401475) q[9];
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
rz(-0.3746364829330034) q[9];
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
rz(0.36636584915544057) q[10];
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
rz(0.3506042430320294) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.01764584168804387) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.37011126251777227) q[11];
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
rz(-0.36807852130624374) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.6906098976549914) q[11];
cx q[10],q[11];
rx(-0.13858472021212326) q[0];
rz(-0.06420591722108672) q[0];
rx(-0.0004401851602195011) q[1];
rz(-0.5287828631968) q[1];
rx(0.00015364026952085363) q[2];
rz(-0.18952222782716532) q[2];
rx(-0.0004234760671066152) q[3];
rz(-0.22858152889039213) q[3];
rx(0.007109026919806421) q[4];
rz(-0.5658653255610496) q[4];
rx(0.000671370176859716) q[5];
rz(-1.4536525587056617) q[5];
rx(-0.000237439710884488) q[6];
rz(-0.26370236216285214) q[6];
rx(-9.718255136084583e-05) q[7];
rz(0.11282879108927185) q[7];
rx(0.0006187313490267871) q[8];
rz(0.15980519638309065) q[8];
rx(-0.000394941670089755) q[9];
rz(0.14564837578107448) q[9];
rx(0.004039737874786467) q[10];
rz(-0.34747095927022015) q[10];
rx(0.0028041328341019407) q[11];
rz(0.09603051776988686) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1364148957357783) q[2];
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
rz(-0.10954727558822648) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.16651533361036827) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.7516052322782905) q[3];
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
rz(-0.7655941655532615) q[3];
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
rz(-0.378308225020287) q[4];
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
rz(-0.3813299662544103) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.04479880119847957) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7765722410410282) q[5];
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
rz(-0.7745662276118599) q[5];
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
rz(-0.09331445526724341) q[6];
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
rz(-0.09026984474754317) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.04328631677832727) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.38262795447642495) q[7];
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
rz(-0.38240231704218025) q[7];
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
rz(-0.7246091882656112) q[8];
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
rz(-0.7122525849367183) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.056867938322991377) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.14518336503022286) q[9];
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
rz(-0.14528480027276236) q[9];
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
rz(-0.0659492749438639) q[10];
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
rz(-0.06796696272345021) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(-0.752498787452883) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.6124895999501696) q[11];
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
rz(-0.6132548484160355) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.061536749956422034) q[11];
cx q[10],q[11];
rx(0.024272524248798794) q[0];
rz(-0.026626298549377106) q[0];
rx(-0.0006178509365465718) q[1];
rz(-0.5371711563602096) q[1];
rx(-0.00012387578294740506) q[2];
rz(0.223899107309149) q[2];
rx(-0.00019032483504368428) q[3];
rz(0.010903643583862501) q[3];
rx(1.8669795635239936e-05) q[4];
rz(0.03729781335819481) q[4];
rx(-0.00021544829035038057) q[5];
rz(-0.22695098513737388) q[5];
rx(-0.00018662769136019379) q[6];
rz(-0.3560099004299367) q[6];
rx(0.0002799716268894535) q[7];
rz(-0.21711649688622528) q[7];
rx(-0.0003891940870690289) q[8];
rz(-0.09255767587882019) q[8];
rx(1.438539048460911e-05) q[9];
rz(-0.18999231546802922) q[9];
rx(-0.0010124599807770328) q[10];
rz(-0.2026591746807182) q[10];
rx(-0.00543782057068863) q[11];
rz(-0.16905234862934318) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.7955605206657829) q[2];
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
rz(-0.7936985174278665) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0564137416691808) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.7947110535957073) q[3];
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
rz(-0.8067741599977449) q[3];
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
rz(-0.31248038247885174) q[4];
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
rz(-0.30833839783514205) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.046797268712954504) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.06720863206537528) q[5];
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
rz(-0.06556199885966235) q[5];
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
rz(-0.20736733020289896) q[6];
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
rz(-0.20731228673676316) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.27232671295898614) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.2569552759875882) q[7];
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
rz(-0.25485776934128723) q[7];
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
rz(0.024801416536108855) q[8];
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
rz(0.023984046991801806) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.027465935249098784) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-0.03433442374075853) q[9];
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
rz(-0.0346082942401353) q[9];
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
rz(-0.9829801882491401) q[10];
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
rz(-0.9822077387990076) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.12576679539579214) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.8282884520415597) q[11];
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
rz(-0.8260040459927285) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.17661624856691455) q[11];
cx q[10],q[11];
rx(0.01598138597060556) q[0];
rz(-0.28553859437910195) q[0];
rx(0.00036066579446566337) q[1];
rz(-0.8585294672515088) q[1];
rx(8.731366855076054e-05) q[2];
rz(-0.1477866445419732) q[2];
rx(0.0001939598725621372) q[3];
rz(-0.1258090240911337) q[3];
rx(1.3919317132427103e-05) q[4];
rz(0.03285384847526793) q[4];
rx(0.0001245586016340664) q[5];
rz(0.08262969066914475) q[5];
rx(8.250490296822055e-05) q[6];
rz(0.06307285927087178) q[6];
rx(-0.00027994517361714185) q[7];
rz(0.06560110157144391) q[7];
rx(0.00039417533215561413) q[8];
rz(0.09719680283022408) q[8];
rx(0.0004529770968041906) q[9];
rz(-0.12379557737508852) q[9];
rx(-0.0009482580741314834) q[10];
rz(0.09552649119055955) q[10];
rx(0.003329283833891877) q[11];
rz(-0.1391700380187052) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.18670508828958102) q[2];
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
rz(-0.19431844348802094) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0712753826153546) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.13049186320075934) q[3];
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
rz(0.13760566057067022) q[3];
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
rz(0.11436389566307528) q[4];
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
rz(0.11387106009157234) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.19606542190076448) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.08218252581709404) q[5];
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
rz(0.0810564248467739) q[5];
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
rz(-0.1640724815164336) q[6];
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
rz(-0.16455055323787235) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.17191184545573523) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.2928862437220276) q[7];
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
rz(0.2915931734313247) q[7];
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
rz(-0.11733492713473503) q[8];
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
rz(-0.11642867899255585) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(0.06530366569514381) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(-1.1289322520003267) q[9];
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
rz(-1.1297845323272846) q[9];
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
rz(-0.39565966384337936) q[10];
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
rz(-0.39635961652434437) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.005511551847173316) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.012921378061696737) q[11];
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
rz(-0.013036249196847894) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.0853225236682642) q[11];
cx q[10],q[11];
rx(-0.009471068553106836) q[0];
rz(0.11760759789141402) q[0];
rx(-0.0002706487963676121) q[1];
rz(0.13541188383224745) q[1];
rx(0.0002458878146998603) q[2];
rz(0.04488622648686185) q[2];
rx(-0.00030161813213598816) q[3];
rz(-0.3330421608031798) q[3];
rx(7.571267925330147e-05) q[4];
rz(-0.042259724558055475) q[4];
rx(0.000265981466741222) q[5];
rz(-0.314942956562721) q[5];
rx(2.92242293413873e-05) q[6];
rz(-0.2425495707717167) q[6];
rx(2.525160199425185e-05) q[7];
rz(-0.13644904848940168) q[7];
rx(5.293765100124628e-05) q[8];
rz(-0.30047508099239256) q[8];
rx(0.00025324336565082196) q[9];
rz(-0.252446022992822) q[9];
rx(0.0014210482828020288) q[10];
rz(-0.22687800654620677) q[10];
rx(0.006199153595379322) q[11];
rz(-0.21933767372684623) q[11];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-1.0081495870368984) q[2];
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
rz(-1.0041277225372487) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.028333189370418944) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-1.0940121646350185) q[3];
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
rz(-1.097031872823149) q[3];
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
rz(-0.478311276517277) q[4];
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
rz(-0.47893961712306693) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.05419982293591528) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.4505521041440622) q[5];
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
rz(-0.44564637380724065) q[5];
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
rz(0.09222863333278415) q[6];
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
rz(0.09383362723103338) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.03870916779586307) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.03140317171301841) q[7];
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
rz(0.031221011168352377) q[7];
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
rz(-0.24016524350087345) q[8];
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
rz(-0.240937005160196) q[8];
cx q[7],q[8];
cx q[6],q[7];
h q[6];
s q[6];
h q[8];
s q[8];
cx q[6],q[7];
rz(-0.00026700351522196056) q[7];
cx q[6],q[7];
h q[7];
h q[9];
cx q[7],q[8];
cx q[8],q[9];
rz(0.796532957317018) q[9];
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
rz(0.7944156127826157) q[9];
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
rz(-0.45953045502557516) q[10];
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
rz(-0.45740431603326437) q[10];
cx q[9],q[10];
cx q[8],q[9];
h q[8];
s q[8];
h q[10];
s q[10];
cx q[8],q[9];
rz(0.01033451950635217) q[9];
cx q[8],q[9];
h q[9];
h q[11];
cx q[9],q[10];
cx q[10],q[11];
rz(-0.3968826378569316) q[11];
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
rz(-0.39533114441688816) q[11];
cx q[10],q[11];
cx q[9],q[10];
h q[9];
s q[9];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.007355844996973967) q[11];
cx q[10],q[11];
rx(0.0031470992168859332) q[0];
rz(-0.19144117741788783) q[0];
rx(0.00023515426602926386) q[1];
rz(0.0032496871611693246) q[1];
rx(0.00023701397845780371) q[2];
rz(-0.11977530324980595) q[2];
rx(-9.729203956760282e-05) q[3];
rz(-0.03128740961475132) q[3];
rx(-0.00015827845757606575) q[4];
rz(-0.21118454362703554) q[4];
rx(-0.0001478912070651741) q[5];
rz(-0.07990224700751836) q[5];
rx(3.0264127915019296e-06) q[6];
rz(0.06510130326292389) q[6];
rx(-4.609210884429866e-05) q[7];
rz(-0.15401105601225037) q[7];
rx(-0.0002610284905854392) q[8];
rz(0.10478739054296791) q[8];
rx(3.47728306420548e-05) q[9];
rz(0.0036460638325923824) q[9];
rx(-0.0003053581906826008) q[10];
rz(0.07127114074519386) q[10];
rx(-0.004840298891022903) q[11];
rz(0.0008922136690376398) q[11];