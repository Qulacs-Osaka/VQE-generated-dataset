OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.05374070249175311) q[0];
ry(0.913497843308802) q[1];
cx q[0],q[1];
ry(-2.4837779787344014) q[0];
ry(2.075869631062086) q[1];
cx q[0],q[1];
ry(2.423244372801339) q[2];
ry(2.6928447557376085) q[3];
cx q[2],q[3];
ry(1.5025023181149422) q[2];
ry(-0.634936300681928) q[3];
cx q[2],q[3];
ry(1.720416526565762) q[0];
ry(-2.021751308730564) q[2];
cx q[0],q[2];
ry(-2.0696123839487326) q[0];
ry(3.0951585892597704) q[2];
cx q[0],q[2];
ry(1.6186649817749326) q[1];
ry(-2.2707943668212787) q[3];
cx q[1],q[3];
ry(-0.4542533516203262) q[1];
ry(2.0423681433540635) q[3];
cx q[1],q[3];
ry(-0.3986309013971943) q[0];
ry(-1.5896885533296792) q[1];
cx q[0],q[1];
ry(0.7594108329020629) q[0];
ry(-0.45614731827771937) q[1];
cx q[0],q[1];
ry(2.7751942098471973) q[2];
ry(2.226275477660188) q[3];
cx q[2],q[3];
ry(2.3916854223093464) q[2];
ry(-0.22473965163626453) q[3];
cx q[2],q[3];
ry(1.9855001743874927) q[0];
ry(0.5897597937895735) q[2];
cx q[0],q[2];
ry(-2.385900096425128) q[0];
ry(2.7457936662986424) q[2];
cx q[0],q[2];
ry(-1.7876388651516097) q[1];
ry(1.8779015071934504) q[3];
cx q[1],q[3];
ry(2.081604445507303) q[1];
ry(0.37609860884656154) q[3];
cx q[1],q[3];
ry(-0.516685654246766) q[0];
ry(-0.8526320885693911) q[1];
cx q[0],q[1];
ry(0.4365171057895906) q[0];
ry(-1.6121421024438218) q[1];
cx q[0],q[1];
ry(1.5404465432804857) q[2];
ry(2.084070172232204) q[3];
cx q[2],q[3];
ry(0.43480206415139655) q[2];
ry(1.2914422404159096) q[3];
cx q[2],q[3];
ry(-1.1420767024414955) q[0];
ry(1.0453941127114685) q[2];
cx q[0],q[2];
ry(-0.10331082056087347) q[0];
ry(-2.769071236138958) q[2];
cx q[0],q[2];
ry(3.07710392591631) q[1];
ry(0.9542727283480783) q[3];
cx q[1],q[3];
ry(-2.145956425049852) q[1];
ry(-0.26013346330924164) q[3];
cx q[1],q[3];
ry(0.5170032772125355) q[0];
ry(-2.4481063448467455) q[1];
cx q[0],q[1];
ry(2.857205489448767) q[0];
ry(1.0139960546822406) q[1];
cx q[0],q[1];
ry(1.2511027224018283) q[2];
ry(2.9810997857199832) q[3];
cx q[2],q[3];
ry(1.3750420504758418) q[2];
ry(0.9345897615985849) q[3];
cx q[2],q[3];
ry(-0.5143747264988816) q[0];
ry(0.3174776998301388) q[2];
cx q[0],q[2];
ry(2.7941076847040085) q[0];
ry(-0.349820491223543) q[2];
cx q[0],q[2];
ry(-3.000300663621724) q[1];
ry(2.1795858814162847) q[3];
cx q[1],q[3];
ry(0.3746307185439157) q[1];
ry(-2.8643110760486823) q[3];
cx q[1],q[3];
ry(-2.9232848282429926) q[0];
ry(2.067430380371257) q[1];
cx q[0],q[1];
ry(1.7644185294191477) q[0];
ry(0.4529022397167512) q[1];
cx q[0],q[1];
ry(2.6217737479508854) q[2];
ry(-0.8093590454638706) q[3];
cx q[2],q[3];
ry(1.0200908704200016) q[2];
ry(0.6393042822370515) q[3];
cx q[2],q[3];
ry(0.7977907388684542) q[0];
ry(1.1860219579821623) q[2];
cx q[0],q[2];
ry(1.858903151809801) q[0];
ry(-0.5747246430126838) q[2];
cx q[0],q[2];
ry(2.9488202836216604) q[1];
ry(0.19701760203615934) q[3];
cx q[1],q[3];
ry(1.683678399546346) q[1];
ry(0.38712197560277684) q[3];
cx q[1],q[3];
ry(-0.7452772569031796) q[0];
ry(-2.7910867964453687) q[1];
cx q[0],q[1];
ry(1.5890658901159194) q[0];
ry(1.877188692574855) q[1];
cx q[0],q[1];
ry(-0.9801554154298877) q[2];
ry(-0.021064895513995208) q[3];
cx q[2],q[3];
ry(-0.8699142966430264) q[2];
ry(-0.5977590490495626) q[3];
cx q[2],q[3];
ry(1.085354513393918) q[0];
ry(2.7906842885164265) q[2];
cx q[0],q[2];
ry(-0.6895251692628959) q[0];
ry(-0.38463537676864407) q[2];
cx q[0],q[2];
ry(2.243656859776727) q[1];
ry(-0.36953528572553784) q[3];
cx q[1],q[3];
ry(1.367879732922301) q[1];
ry(2.600193515121269) q[3];
cx q[1],q[3];
ry(0.42096261648206745) q[0];
ry(-0.07358347592313158) q[1];
ry(2.5897909423491043) q[2];
ry(2.579956323380812) q[3];