OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.8876636742667441) q[0];
ry(2.2172265301265126) q[1];
cx q[0],q[1];
ry(2.0350663692894244) q[0];
ry(-2.7958673239243517) q[1];
cx q[0],q[1];
ry(0.6459496920363206) q[1];
ry(-2.942716503202008) q[2];
cx q[1],q[2];
ry(1.9619071881734576) q[1];
ry(-2.699487632916335) q[2];
cx q[1],q[2];
ry(-1.722509193295) q[2];
ry(2.651606955855255) q[3];
cx q[2],q[3];
ry(2.175966130302366) q[2];
ry(0.9080119394611285) q[3];
cx q[2],q[3];
ry(-0.09294237104124954) q[0];
ry(-1.202713474674856) q[1];
cx q[0],q[1];
ry(-2.1012309715291733) q[0];
ry(1.4074325913798722) q[1];
cx q[0],q[1];
ry(0.5952235197185471) q[1];
ry(0.9345229584565835) q[2];
cx q[1],q[2];
ry(-0.5489453100413698) q[1];
ry(2.7701157537179055) q[2];
cx q[1],q[2];
ry(-0.7506940767302022) q[2];
ry(-0.7567040632664837) q[3];
cx q[2],q[3];
ry(-2.4555156455353178) q[2];
ry(3.018614700148508) q[3];
cx q[2],q[3];
ry(3.018117276718146) q[0];
ry(-0.29317156891686463) q[1];
cx q[0],q[1];
ry(-2.6409922866657083) q[0];
ry(1.878387046478612) q[1];
cx q[0],q[1];
ry(1.8060420815641758) q[1];
ry(1.398223544519639) q[2];
cx q[1],q[2];
ry(0.413667576374988) q[1];
ry(2.4816751379807096) q[2];
cx q[1],q[2];
ry(1.1709745142558772) q[2];
ry(2.1181982001593695) q[3];
cx q[2],q[3];
ry(-0.5457771814566241) q[2];
ry(0.9079336119779873) q[3];
cx q[2],q[3];
ry(-0.48127651556605944) q[0];
ry(-2.7249129095043916) q[1];
cx q[0],q[1];
ry(-0.8915318305669736) q[0];
ry(-0.8033789821746457) q[1];
cx q[0],q[1];
ry(2.9598762185259084) q[1];
ry(2.128635963622017) q[2];
cx q[1],q[2];
ry(-2.722153434798455) q[1];
ry(-1.8979078502387783) q[2];
cx q[1],q[2];
ry(-2.3529081140819197) q[2];
ry(0.01665574645893031) q[3];
cx q[2],q[3];
ry(0.6402888996286693) q[2];
ry(0.07039352727994254) q[3];
cx q[2],q[3];
ry(0.9910279900761384) q[0];
ry(1.2019817852069998) q[1];
cx q[0],q[1];
ry(-2.5181621934470635) q[0];
ry(-2.881618734249219) q[1];
cx q[0],q[1];
ry(-0.061206700115092896) q[1];
ry(1.8627386308739586) q[2];
cx q[1],q[2];
ry(0.6811731386911266) q[1];
ry(-2.7123088702624276) q[2];
cx q[1],q[2];
ry(1.0028454652687593) q[2];
ry(0.2281306420145817) q[3];
cx q[2],q[3];
ry(2.7783830563792913) q[2];
ry(-0.8098114708287698) q[3];
cx q[2],q[3];
ry(2.5189100534245363) q[0];
ry(-1.0251131414540406) q[1];
cx q[0],q[1];
ry(-2.145823166888918) q[0];
ry(1.295915789616855) q[1];
cx q[0],q[1];
ry(-2.9267359689654735) q[1];
ry(-1.2426021810767396) q[2];
cx q[1],q[2];
ry(2.4623696643316135) q[1];
ry(-2.418314956051228) q[2];
cx q[1],q[2];
ry(-0.21543054747555399) q[2];
ry(1.0141697394417983) q[3];
cx q[2],q[3];
ry(0.4589469154207446) q[2];
ry(-1.577498920613123) q[3];
cx q[2],q[3];
ry(-2.212651836961623) q[0];
ry(-1.428738800571905) q[1];
cx q[0],q[1];
ry(0.47111456132047036) q[0];
ry(-2.7656540144186255) q[1];
cx q[0],q[1];
ry(1.1192214835669647) q[1];
ry(0.535879677198186) q[2];
cx q[1],q[2];
ry(1.4817495088828536) q[1];
ry(2.3533564607354016) q[2];
cx q[1],q[2];
ry(-0.6735178901925705) q[2];
ry(0.40377904442431145) q[3];
cx q[2],q[3];
ry(0.2702413219026471) q[2];
ry(1.8599694466755472) q[3];
cx q[2],q[3];
ry(2.9737031275771644) q[0];
ry(-2.5415409203454478) q[1];
cx q[0],q[1];
ry(-2.2741229355732706) q[0];
ry(-0.39748159328991495) q[1];
cx q[0],q[1];
ry(-2.210219077910226) q[1];
ry(-2.3884195790196787) q[2];
cx q[1],q[2];
ry(0.9890268414742649) q[1];
ry(-0.6576751219914883) q[2];
cx q[1],q[2];
ry(1.6324088811751505) q[2];
ry(1.6084862613037298) q[3];
cx q[2],q[3];
ry(3.0626009859787637) q[2];
ry(-0.8359184527392403) q[3];
cx q[2],q[3];
ry(1.5694782335772028) q[0];
ry(1.3758051170741723) q[1];
ry(0.42273933201968694) q[2];
ry(1.5456129300000283) q[3];