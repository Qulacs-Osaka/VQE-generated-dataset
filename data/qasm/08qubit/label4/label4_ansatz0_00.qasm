OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03357830650018927) q[2];
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
rz(-0.038404079365073424) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.621672233796491) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.5387535609251376) q[3];
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
rz(0.4303681451520013) q[3];
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
rz(-0.08022290011822776) q[4];
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
rz(0.3423439814641904) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-1.8663959628169768) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(2.22596377818401) q[5];
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
rz(0.12827309364029985) q[5];
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
rz(-1.3421329348922642) q[6];
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
rz(0.8858178392232772) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.07718562462045835) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.0229916999318538) q[7];
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
rz(-1.2012124103809687) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.00783382299988191) q[7];
cx q[6],q[7];
rx(0.01869170934885261) q[0];
rz(1.9005169816727645) q[0];
rx(0.0004548138997196692) q[1];
rz(0.04926268964569113) q[1];
rx(-1.938214083739546) q[2];
rz(0.00043021930715789316) q[2];
rx(-3.1414855772934707) q[3];
rz(-3.0299478866754814) q[3];
rx(-0.00017132087236514085) q[4];
rz(0.04012248795197272) q[4];
rx(3.1415068918928393) q[5];
rz(0.05775944230108768) q[5];
rx(-3.7880768489439277e-06) q[6];
rz(-0.024674896355433042) q[6];
rx(-0.0003115005299997691) q[7];
rz(0.015192555975618053) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(1.5721634945696357) q[2];
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
rz(-1.4407905967069645) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-1.5709521089180087) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(1.2926486589254769) q[3];
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
rz(-1.597796876884126) q[3];
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
rz(0.1232142527551225) q[4];
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
rz(-0.4061699969520526) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.3213461987046799) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(1.2777382847445218) q[5];
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
rz(1.588834803452306) q[5];
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
rz(-0.6734229864436956) q[6];
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
rz(-0.013828288652214017) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.22173150283001386) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.9516515965184424) q[7];
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
rz(1.101242005039229) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-2.831064488999554) q[7];
cx q[6],q[7];
rx(-1.9410907882064559) q[0];
rz(-0.016260049252020115) q[0];
rx(-6.870787932957796e-05) q[1];
rz(-2.9988590863586566) q[1];
rx(-3.1414458046354556) q[2];
rz(-0.03162328376141432) q[2];
rx(3.1414767871378384) q[3];
rz(1.634411015072355) q[3];
rx(3.141498604580924) q[4];
rz(-0.5453650672852226) q[4];
rx(-3.141536498809585) q[5];
rz(1.5737871587760752) q[5];
rx(-0.0001281849644733333) q[6];
rz(-2.566499025304953) q[6];
rx(2.022223740510235) q[7];
rz(0.0003062394972013248) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-2.4855497314627515) q[2];
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
rz(0.8462422680273692) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.27782801084080844) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0020050999790333216) q[3];
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
rz(-1.562017379143663) q[3];
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
rz(0.04726877977828008) q[4];
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
rz(0.04143464644054045) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.00023383697441146787) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0006320400077884543) q[5];
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
rz(0.6799528864286938) q[5];
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
rz(-0.001142475710618263) q[6];
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
rz(-0.011109819007575579) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.0007369807690081037) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(1.5833680209288938) q[7];
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
rz(8.072137829248332e-05) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-3.1415558362125866) q[7];
cx q[6],q[7];
rx(0.0047228284780453395) q[0];
rz(-0.8675749072476016) q[0];
rx(3.266658163458468e-05) q[1];
rz(-0.14465758788000765) q[1];
rx(1.7666476857195352e-05) q[2];
rz(-2.4227578172238484) q[2];
rx(3.1414848536783517) q[3];
rz(0.26874737735854204) q[3];
rx(0.00030101675029255555) q[4];
rz(-0.2759568743196521) q[4];
rx(-3.1415914217618424) q[5];
rz(-0.6646167463680596) q[5];
rx(-2.264841563939174e-05) q[6];
rz(-2.867894347698231) q[6];
rx(2.0216245170394247) q[7];
rz(0.4116690391303647) q[7];