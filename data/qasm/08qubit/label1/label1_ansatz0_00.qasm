OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.4564727279987414) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.32636010237866525) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07857198847782744) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.0959738267705736) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.8664129795190599) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5772949697508225) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.7058392693721068) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3234103014019171) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05934844638893922) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(2.2082855430399824) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-2.2101980711903275) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(1.212966807066705) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(3.013943140401667) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-3.0139784545875092) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(3.017289609882179) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.823961580364817) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.3194612645273989) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.8365779663848384) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.15923862936557154) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.1590542459268958) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.5896110576759728) q[7];
cx q[6],q[7];
rz(0.5817064379335036) q[0];
rz(0.6378346539358744) q[1];
rz(0.8893597110345626) q[2];
rz(-2.134016718755145) q[3];
rz(-2.061388729363365) q[4];
rz(-0.28029598982936327) q[5];
rz(0.07789184371422402) q[6];
rz(0.8500286544155048) q[7];
rx(-0.00013764782082109088) q[0];
rx(-0.00021674555333474097) q[1];
rx(-3.1414524807518487) q[2];
rx(-3.1414987725496406) q[3];
rx(-4.748896450991389e-05) q[4];
rx(-3.141392987408763) q[5];
rx(0.00011750522420972268) q[6];
rx(3.1415796843889034) q[7];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.17869143516642005) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(2.373824356897975) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-2.778410632482581) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.5990971417836527) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.3437539050384506) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.6834501746456773) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.19761886103397414) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.19772179861423927) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13628207887266855) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-2.934241424942817) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-2.9363460362540676) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.12679324582053692) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.22311509595600357) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.22300041400676107) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.23462315716059812) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.43204164688593066) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.4317415362433412) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.3586955470686395) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-2.683828666910387) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.4576763454896989) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.3638249661075937) q[7];
cx q[6],q[7];
rz(-1.2051414059469212) q[0];
rz(1.3360510811910302) q[1];
rz(1.105636870585826) q[2];
rz(-1.0340717040346763) q[3];
rz(-1.4445689852308963) q[4];
rz(2.393495742697779) q[5];
rz(-2.403745054888007) q[6];
rz(-0.5263392258388477) q[7];
rx(-0.00013530165051913343) q[0];
rx(4.892921644705935e-05) q[1];
rx(7.49498178151542e-05) q[2];
rx(6.422037000179603e-06) q[3];
rx(3.530662326943006e-05) q[4];
rx(-3.141384007332069) q[5];
rx(8.978873209846695e-05) q[6];
rx(-0.00013467620108961647) q[7];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.29660102210882344) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.011495937146126053) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.5228312712368796) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.9701075033782457) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.9687074743670252) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-1.127847544339732) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.36392173328925165) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3635257111906524) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07260973000713364) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(2.5921663701055344) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(2.591315910155526) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.4369793139735832) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.5934895658547087) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.5936848837156893) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.42532571487695825) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.6213578117766043) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.6214083095182451) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.5628039948162722) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.6987584734305151) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-2.4428500435330203) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.5561584300928555) q[7];
cx q[6],q[7];
rz(-0.01933497677342741) q[0];
rz(-0.20239834919241215) q[1];
rz(-0.688255268996962) q[2];
rz(0.7487354648317472) q[3];
rz(0.7893867523229435) q[4];
rz(-1.7989296019117667) q[5];
rz(-1.4454305566295012) q[6];
rz(0.6440191456677975) q[7];
rx(0.00036261158986785865) q[0];
rx(0.00023302822555539086) q[1];
rx(3.1415601668344046) q[2];
rx(3.141451329579071) q[3];
rx(-3.1415747082636085) q[4];
rx(-3.141490151993738) q[5];
rx(8.273938727481243e-05) q[6];
rx(-9.401645741933731e-05) q[7];