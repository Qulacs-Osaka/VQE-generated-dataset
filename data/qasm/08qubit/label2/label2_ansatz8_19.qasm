OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.501492121151987) q[0];
ry(2.0055520431313054) q[1];
cx q[0],q[1];
ry(-1.8230481981617963) q[0];
ry(-3.0367873689956877) q[1];
cx q[0],q[1];
ry(2.0717606998082805) q[2];
ry(1.6572594427010374) q[3];
cx q[2],q[3];
ry(1.6117637027290899) q[2];
ry(0.33695347350064253) q[3];
cx q[2],q[3];
ry(-0.8611343007616564) q[4];
ry(1.5317941417025285) q[5];
cx q[4],q[5];
ry(-2.935844731014299) q[4];
ry(0.14008155205505357) q[5];
cx q[4],q[5];
ry(3.0706856641095603) q[6];
ry(-0.2900728859811492) q[7];
cx q[6],q[7];
ry(-1.3850732025264971) q[6];
ry(2.8244351405169823) q[7];
cx q[6],q[7];
ry(0.5007368222596299) q[0];
ry(-1.9776917562908514) q[2];
cx q[0],q[2];
ry(0.7583573501100345) q[0];
ry(-1.5776117939628524) q[2];
cx q[0],q[2];
ry(-0.7641903736206542) q[2];
ry(1.9393887069095808) q[4];
cx q[2],q[4];
ry(-2.275819425537022) q[2];
ry(1.541479975076852) q[4];
cx q[2],q[4];
ry(2.145734763917767) q[4];
ry(-0.189388107215656) q[6];
cx q[4],q[6];
ry(-0.8720488980858198) q[4];
ry(2.464274887865475) q[6];
cx q[4],q[6];
ry(-2.680583028014473) q[1];
ry(2.041301231458696) q[3];
cx q[1],q[3];
ry(1.5243825566697948) q[1];
ry(-1.1725445026310526) q[3];
cx q[1],q[3];
ry(-1.269322420824634) q[3];
ry(-1.5231503405325852) q[5];
cx q[3],q[5];
ry(-1.4292448175728019) q[3];
ry(2.5093407247544084) q[5];
cx q[3],q[5];
ry(-2.6349708656221624) q[5];
ry(1.4474995940701207) q[7];
cx q[5],q[7];
ry(2.604898244594969) q[5];
ry(0.7179796240073583) q[7];
cx q[5],q[7];
ry(0.7846357730229787) q[0];
ry(-0.274525265615563) q[1];
cx q[0],q[1];
ry(1.3047695222824818) q[0];
ry(2.0743314422143047) q[1];
cx q[0],q[1];
ry(-0.24205987801313433) q[2];
ry(-2.116619954007333) q[3];
cx q[2],q[3];
ry(2.5750032302054198) q[2];
ry(-1.9297773056841168) q[3];
cx q[2],q[3];
ry(2.943878704435623) q[4];
ry(-1.7346226597399648) q[5];
cx q[4],q[5];
ry(-0.3674145370149986) q[4];
ry(-1.2721668007073852) q[5];
cx q[4],q[5];
ry(1.8544366046570384) q[6];
ry(0.2060330074412208) q[7];
cx q[6],q[7];
ry(3.1173131514422967) q[6];
ry(-0.9263687763294409) q[7];
cx q[6],q[7];
ry(2.9980756692164428) q[0];
ry(3.000912555259611) q[2];
cx q[0],q[2];
ry(-1.661248133756768) q[0];
ry(-1.731796587631703) q[2];
cx q[0],q[2];
ry(1.0058999196535257) q[2];
ry(-2.943005056516375) q[4];
cx q[2],q[4];
ry(0.48993074214470594) q[2];
ry(0.38299558321609983) q[4];
cx q[2],q[4];
ry(1.6308990734218833) q[4];
ry(1.3745383234797772) q[6];
cx q[4],q[6];
ry(-1.291195316901454) q[4];
ry(-0.3436750786867025) q[6];
cx q[4],q[6];
ry(1.5320425984503512) q[1];
ry(2.977819103614888) q[3];
cx q[1],q[3];
ry(-0.9468335241627338) q[1];
ry(2.327331206667885) q[3];
cx q[1],q[3];
ry(3.114288385420607) q[3];
ry(-2.373306620430914) q[5];
cx q[3],q[5];
ry(2.7345320130777915) q[3];
ry(-0.06525513262041838) q[5];
cx q[3],q[5];
ry(-2.877137162318086) q[5];
ry(-1.1332569200183091) q[7];
cx q[5],q[7];
ry(1.843329550989658) q[5];
ry(-1.4176090029662662) q[7];
cx q[5],q[7];
ry(-0.3164831998454141) q[0];
ry(-1.9688058192503028) q[1];
cx q[0],q[1];
ry(-1.1692512564905364) q[0];
ry(2.466842464645416) q[1];
cx q[0],q[1];
ry(1.6041859989759262) q[2];
ry(2.1192864398321998) q[3];
cx q[2],q[3];
ry(2.988062881411371) q[2];
ry(2.6955232466610775) q[3];
cx q[2],q[3];
ry(-1.0396915736489527) q[4];
ry(1.020225750522508) q[5];
cx q[4],q[5];
ry(2.9169687189079982) q[4];
ry(2.9980911915557544) q[5];
cx q[4],q[5];
ry(-2.8019375620132747) q[6];
ry(2.861576308266928) q[7];
cx q[6],q[7];
ry(-1.5838478157247653) q[6];
ry(2.1539548040892553) q[7];
cx q[6],q[7];
ry(-1.040186364928802) q[0];
ry(-2.57861409461185) q[2];
cx q[0],q[2];
ry(-2.5709311223347875) q[0];
ry(-0.24032686335372144) q[2];
cx q[0],q[2];
ry(-3.0715555986605407) q[2];
ry(1.3437491911600017) q[4];
cx q[2],q[4];
ry(0.860679030413263) q[2];
ry(1.8322100007395263) q[4];
cx q[2],q[4];
ry(1.408828029388084) q[4];
ry(2.836004924616926) q[6];
cx q[4],q[6];
ry(2.9700472016031823) q[4];
ry(-1.582785291174483) q[6];
cx q[4],q[6];
ry(-2.5021134386043813) q[1];
ry(2.5531820001677095) q[3];
cx q[1],q[3];
ry(-2.670649843063502) q[1];
ry(-0.9430228508838016) q[3];
cx q[1],q[3];
ry(-0.4339355074980609) q[3];
ry(2.2819325104290247) q[5];
cx q[3],q[5];
ry(0.7939451729910934) q[3];
ry(0.7496328609047739) q[5];
cx q[3],q[5];
ry(-2.35794507847099) q[5];
ry(-2.155380062136487) q[7];
cx q[5],q[7];
ry(1.4549111156358605) q[5];
ry(-0.4038146119603742) q[7];
cx q[5],q[7];
ry(-3.1011923827080694) q[0];
ry(2.7119498715779744) q[1];
cx q[0],q[1];
ry(0.33050479151192746) q[0];
ry(-2.8894151969116124) q[1];
cx q[0],q[1];
ry(-2.819924924253616) q[2];
ry(-0.04083431959732273) q[3];
cx q[2],q[3];
ry(1.7962184383472648) q[2];
ry(0.6798804065768183) q[3];
cx q[2],q[3];
ry(-0.12817517589502092) q[4];
ry(1.5610056408699686) q[5];
cx q[4],q[5];
ry(-1.694486861660871) q[4];
ry(2.1235067943992174) q[5];
cx q[4],q[5];
ry(-0.5566485558657783) q[6];
ry(1.471400567423423) q[7];
cx q[6],q[7];
ry(2.6870652167534126) q[6];
ry(2.261499931557886) q[7];
cx q[6],q[7];
ry(3.037550314702642) q[0];
ry(-2.9916720289791554) q[2];
cx q[0],q[2];
ry(-2.501732740640111) q[0];
ry(2.373194890965611) q[2];
cx q[0],q[2];
ry(0.43160157885357875) q[2];
ry(2.06407444957887) q[4];
cx q[2],q[4];
ry(-0.028010592285030004) q[2];
ry(-0.18092530617147382) q[4];
cx q[2],q[4];
ry(0.9782579330071668) q[4];
ry(-2.3972615604094067) q[6];
cx q[4],q[6];
ry(-2.3895653124723277) q[4];
ry(1.2687620012370955) q[6];
cx q[4],q[6];
ry(-0.9893486321733169) q[1];
ry(2.2759728888528175) q[3];
cx q[1],q[3];
ry(-1.9712943045025426) q[1];
ry(2.626408641051721) q[3];
cx q[1],q[3];
ry(-2.103147852917191) q[3];
ry(2.480919678754346) q[5];
cx q[3],q[5];
ry(-3.0250726053030284) q[3];
ry(-2.053893896657087) q[5];
cx q[3],q[5];
ry(-2.4663438460255946) q[5];
ry(-2.091085997028311) q[7];
cx q[5],q[7];
ry(1.3651181537581853) q[5];
ry(-2.3945413294948996) q[7];
cx q[5],q[7];
ry(2.5643334430244016) q[0];
ry(-1.6500264119057082) q[1];
cx q[0],q[1];
ry(1.8905141788821913) q[0];
ry(3.105577030127943) q[1];
cx q[0],q[1];
ry(1.5275161153006866) q[2];
ry(-1.591345200798104) q[3];
cx q[2],q[3];
ry(-1.7547021248684527) q[2];
ry(-2.867938242361992) q[3];
cx q[2],q[3];
ry(2.554688602005687) q[4];
ry(-2.876073425870023) q[5];
cx q[4],q[5];
ry(-1.6659207277682404) q[4];
ry(1.6010376133197455) q[5];
cx q[4],q[5];
ry(0.5183071366680698) q[6];
ry(2.174497798015028) q[7];
cx q[6],q[7];
ry(-1.8608702186111756) q[6];
ry(-0.7588843789259823) q[7];
cx q[6],q[7];
ry(-1.8111461546419267) q[0];
ry(-0.4915893354489551) q[2];
cx q[0],q[2];
ry(-0.034857675433773494) q[0];
ry(0.1054941272455795) q[2];
cx q[0],q[2];
ry(-0.17548490974298758) q[2];
ry(-0.014950487625082488) q[4];
cx q[2],q[4];
ry(-2.5586805912634447) q[2];
ry(1.4496397250133566) q[4];
cx q[2],q[4];
ry(3.106060052830419) q[4];
ry(2.7321181649895303) q[6];
cx q[4],q[6];
ry(1.292048234000221) q[4];
ry(-2.466435092281965) q[6];
cx q[4],q[6];
ry(1.6811474791363625) q[1];
ry(0.8886955918919535) q[3];
cx q[1],q[3];
ry(0.037051544326928344) q[1];
ry(-3.0132588385847736) q[3];
cx q[1],q[3];
ry(-0.012208017875588341) q[3];
ry(-1.6347732861621536) q[5];
cx q[3],q[5];
ry(1.4704336952334494) q[3];
ry(-1.5479816212023405) q[5];
cx q[3],q[5];
ry(2.1340228618527295) q[5];
ry(-0.538486793965455) q[7];
cx q[5],q[7];
ry(-0.8723036074810109) q[5];
ry(-1.0978727861943183) q[7];
cx q[5],q[7];
ry(-0.36710289340898505) q[0];
ry(1.9408089313567292) q[1];
cx q[0],q[1];
ry(-2.0541864418653892) q[0];
ry(1.9182604466417523) q[1];
cx q[0],q[1];
ry(-2.115881147613853) q[2];
ry(-2.6603130657198424) q[3];
cx q[2],q[3];
ry(-2.2507486062663835) q[2];
ry(-0.9438618766776588) q[3];
cx q[2],q[3];
ry(2.5796265002512815) q[4];
ry(2.125612420580489) q[5];
cx q[4],q[5];
ry(-2.645429317757045) q[4];
ry(-0.40496211513621105) q[5];
cx q[4],q[5];
ry(-2.3270910843302848) q[6];
ry(-2.378367584509207) q[7];
cx q[6],q[7];
ry(0.41540170816679195) q[6];
ry(-0.35307174641043293) q[7];
cx q[6],q[7];
ry(0.8063992416604635) q[0];
ry(-0.5746703085725509) q[2];
cx q[0],q[2];
ry(2.025704396459793) q[0];
ry(-0.15119219809895504) q[2];
cx q[0],q[2];
ry(-1.819168628923207) q[2];
ry(1.0804286399624634) q[4];
cx q[2],q[4];
ry(0.3807383053546438) q[2];
ry(0.9418815906256203) q[4];
cx q[2],q[4];
ry(2.355090058015313) q[4];
ry(2.6626331325372163) q[6];
cx q[4],q[6];
ry(-2.563407695563335) q[4];
ry(-0.5329219100321425) q[6];
cx q[4],q[6];
ry(0.8035745399059837) q[1];
ry(-0.21694396933566143) q[3];
cx q[1],q[3];
ry(-2.244219435807121) q[1];
ry(1.405907748160556) q[3];
cx q[1],q[3];
ry(0.13747705104589253) q[3];
ry(-3.0037876648441957) q[5];
cx q[3],q[5];
ry(-0.763552306200749) q[3];
ry(2.3100593195483965) q[5];
cx q[3],q[5];
ry(-2.81609047067519) q[5];
ry(-3.1262102134246095) q[7];
cx q[5],q[7];
ry(-0.8769965059662583) q[5];
ry(-0.36497905410579623) q[7];
cx q[5],q[7];
ry(1.525819100291773) q[0];
ry(0.16515042306736125) q[1];
cx q[0],q[1];
ry(0.7639504615114548) q[0];
ry(2.6301563238455037) q[1];
cx q[0],q[1];
ry(0.42426476757506837) q[2];
ry(2.072873778684434) q[3];
cx q[2],q[3];
ry(2.940309489908158) q[2];
ry(1.9133453569766266) q[3];
cx q[2],q[3];
ry(-0.6817651252695089) q[4];
ry(1.3370140478675383) q[5];
cx q[4],q[5];
ry(1.989550565092981) q[4];
ry(0.31796562156336383) q[5];
cx q[4],q[5];
ry(0.8597120969376801) q[6];
ry(0.5983818007868917) q[7];
cx q[6],q[7];
ry(-1.0527021386798028) q[6];
ry(2.8373430575932788) q[7];
cx q[6],q[7];
ry(-1.5839144366260651) q[0];
ry(-1.2638187894820874) q[2];
cx q[0],q[2];
ry(2.213311486337779) q[0];
ry(-1.8457912729997177) q[2];
cx q[0],q[2];
ry(-3.097769690017394) q[2];
ry(-2.41907383856587) q[4];
cx q[2],q[4];
ry(-3.074983699617052) q[2];
ry(-0.8361867299091313) q[4];
cx q[2],q[4];
ry(-1.564165901483622) q[4];
ry(1.0322735438205077) q[6];
cx q[4],q[6];
ry(2.556186881523724) q[4];
ry(-1.8092182450944307) q[6];
cx q[4],q[6];
ry(-2.3657012433535005) q[1];
ry(-1.330677726485575) q[3];
cx q[1],q[3];
ry(1.8807275112291832) q[1];
ry(3.0280212292768116) q[3];
cx q[1],q[3];
ry(2.1499910844899235) q[3];
ry(0.2947395678067153) q[5];
cx q[3],q[5];
ry(-1.5552414069647167) q[3];
ry(-1.0090666578257024) q[5];
cx q[3],q[5];
ry(-1.7108259358905769) q[5];
ry(-2.156569572236133) q[7];
cx q[5],q[7];
ry(2.1100867358392774) q[5];
ry(-0.7871013859840987) q[7];
cx q[5],q[7];
ry(-2.769534050813084) q[0];
ry(-1.0599092514957658) q[1];
cx q[0],q[1];
ry(-2.022696330236716) q[0];
ry(1.0626241184072034) q[1];
cx q[0],q[1];
ry(-2.028176068381618) q[2];
ry(-0.9750845116510689) q[3];
cx q[2],q[3];
ry(2.536488827681484) q[2];
ry(3.079465555250235) q[3];
cx q[2],q[3];
ry(2.212183024860909) q[4];
ry(2.4490110921318298) q[5];
cx q[4],q[5];
ry(1.6234877251688253) q[4];
ry(2.301238486009822) q[5];
cx q[4],q[5];
ry(2.9541839614496097) q[6];
ry(-2.4869812499429482) q[7];
cx q[6],q[7];
ry(0.9448290670166257) q[6];
ry(-2.3477023116061386) q[7];
cx q[6],q[7];
ry(-0.8701654749305453) q[0];
ry(0.30805365870656315) q[2];
cx q[0],q[2];
ry(2.6174801710893174) q[0];
ry(-2.816797030342095) q[2];
cx q[0],q[2];
ry(2.4379672642936985) q[2];
ry(-1.3040230258650232) q[4];
cx q[2],q[4];
ry(-2.8106866193407742) q[2];
ry(-2.1244835875412225) q[4];
cx q[2],q[4];
ry(-2.717055715025243) q[4];
ry(-3.1212428934136827) q[6];
cx q[4],q[6];
ry(-1.7585845358701508) q[4];
ry(-2.3551209579266015) q[6];
cx q[4],q[6];
ry(1.6246510202013198) q[1];
ry(-2.5739613053633534) q[3];
cx q[1],q[3];
ry(2.436643090751681) q[1];
ry(-2.5569791614031026) q[3];
cx q[1],q[3];
ry(1.0150608437240018) q[3];
ry(1.1295123218744456) q[5];
cx q[3],q[5];
ry(-1.6297048300637869) q[3];
ry(-2.912217879247596) q[5];
cx q[3],q[5];
ry(-0.6893754547482691) q[5];
ry(2.4502707107430926) q[7];
cx q[5],q[7];
ry(1.494810866074434) q[5];
ry(-2.0359787420160513) q[7];
cx q[5],q[7];
ry(2.323581055585777) q[0];
ry(1.152502804047824) q[1];
cx q[0],q[1];
ry(1.322319349224693) q[0];
ry(0.9321079774930086) q[1];
cx q[0],q[1];
ry(-1.8469186875586772) q[2];
ry(1.247708435794381) q[3];
cx q[2],q[3];
ry(-1.5832374297366274) q[2];
ry(-2.8024907229329843) q[3];
cx q[2],q[3];
ry(2.938613034913516) q[4];
ry(2.9457310784604602) q[5];
cx q[4],q[5];
ry(-2.7177978989340255) q[4];
ry(-0.14942989308138968) q[5];
cx q[4],q[5];
ry(1.7828581970781527) q[6];
ry(-1.8128914603931892) q[7];
cx q[6],q[7];
ry(-2.75634513263574) q[6];
ry(3.031427555350199) q[7];
cx q[6],q[7];
ry(1.1184795877478086) q[0];
ry(1.6744214548945324) q[2];
cx q[0],q[2];
ry(0.24612813552314614) q[0];
ry(1.603379768185647) q[2];
cx q[0],q[2];
ry(0.8852807216147857) q[2];
ry(-2.507951510684968) q[4];
cx q[2],q[4];
ry(2.8302853388921556) q[2];
ry(2.8355386970648273) q[4];
cx q[2],q[4];
ry(1.7705004379002485) q[4];
ry(1.863802669406775) q[6];
cx q[4],q[6];
ry(0.9353812142737379) q[4];
ry(-0.15943889000630854) q[6];
cx q[4],q[6];
ry(1.4377130924575579) q[1];
ry(-3.039363422321727) q[3];
cx q[1],q[3];
ry(-1.2313349780867409) q[1];
ry(0.15096575451235328) q[3];
cx q[1],q[3];
ry(-2.8274159246859414) q[3];
ry(0.26773607845551206) q[5];
cx q[3],q[5];
ry(2.368834554862994) q[3];
ry(-2.3104224749775613) q[5];
cx q[3],q[5];
ry(3.059159032275435) q[5];
ry(2.881485438939558) q[7];
cx q[5],q[7];
ry(-0.14984329600535434) q[5];
ry(1.7630729947067962) q[7];
cx q[5],q[7];
ry(2.126637900806998) q[0];
ry(0.8703593636607507) q[1];
cx q[0],q[1];
ry(-1.1633015326599678) q[0];
ry(0.00407920200248224) q[1];
cx q[0],q[1];
ry(-1.320072720694632) q[2];
ry(-1.1178872484682278) q[3];
cx q[2],q[3];
ry(-2.3319258361442805) q[2];
ry(-0.1285340896146969) q[3];
cx q[2],q[3];
ry(-0.94110084904323) q[4];
ry(3.032297540955643) q[5];
cx q[4],q[5];
ry(-1.690992214816621) q[4];
ry(-0.7763151015347693) q[5];
cx q[4],q[5];
ry(-2.859040233679617) q[6];
ry(0.7357917009814324) q[7];
cx q[6],q[7];
ry(-0.6871813685648871) q[6];
ry(-1.3946033532399675) q[7];
cx q[6],q[7];
ry(2.4897771290504176) q[0];
ry(-2.701979402323427) q[2];
cx q[0],q[2];
ry(2.601598018881332) q[0];
ry(-0.5744933148131812) q[2];
cx q[0],q[2];
ry(-1.2062286222020338) q[2];
ry(-1.8796571083789133) q[4];
cx q[2],q[4];
ry(0.7150519031006306) q[2];
ry(-0.469943911720896) q[4];
cx q[2],q[4];
ry(0.034584026501564225) q[4];
ry(-0.6818980546236572) q[6];
cx q[4],q[6];
ry(2.0379158543782045) q[4];
ry(-1.7674593066730981) q[6];
cx q[4],q[6];
ry(2.49398275875187) q[1];
ry(1.1912432285609806) q[3];
cx q[1],q[3];
ry(1.3815736792222817) q[1];
ry(0.48250711791058176) q[3];
cx q[1],q[3];
ry(-0.14568805788092032) q[3];
ry(1.6124693292107095) q[5];
cx q[3],q[5];
ry(-0.506249844684384) q[3];
ry(-0.21563662004461462) q[5];
cx q[3],q[5];
ry(0.8865161080199234) q[5];
ry(-2.3814019979819836) q[7];
cx q[5],q[7];
ry(3.004380991199062) q[5];
ry(1.9152690337110663) q[7];
cx q[5],q[7];
ry(1.8036870235108537) q[0];
ry(-2.1914155925743035) q[1];
cx q[0],q[1];
ry(-2.3617494970017225) q[0];
ry(1.6986071889508014) q[1];
cx q[0],q[1];
ry(0.8797911926511384) q[2];
ry(0.5856206382848391) q[3];
cx q[2],q[3];
ry(-0.8553107786627772) q[2];
ry(1.608651798530496) q[3];
cx q[2],q[3];
ry(0.8445528988089624) q[4];
ry(1.9105131822637929) q[5];
cx q[4],q[5];
ry(3.072061853799061) q[4];
ry(2.2518326883587303) q[5];
cx q[4],q[5];
ry(2.4205070780823976) q[6];
ry(-2.579827294878252) q[7];
cx q[6],q[7];
ry(1.6502280765096993) q[6];
ry(2.054798772341603) q[7];
cx q[6],q[7];
ry(-2.9284178113119164) q[0];
ry(0.9918595497430553) q[2];
cx q[0],q[2];
ry(-2.237395085577164) q[0];
ry(-2.372263121095766) q[2];
cx q[0],q[2];
ry(0.25606660125352204) q[2];
ry(-1.1552761024589797) q[4];
cx q[2],q[4];
ry(1.7222236862124198) q[2];
ry(0.7273747821483534) q[4];
cx q[2],q[4];
ry(1.6020154922797367) q[4];
ry(-2.6642538598886647) q[6];
cx q[4],q[6];
ry(-0.7506968101133487) q[4];
ry(-1.9516255420441901) q[6];
cx q[4],q[6];
ry(1.2398432552588947) q[1];
ry(0.7996528786660758) q[3];
cx q[1],q[3];
ry(-1.517585584420758) q[1];
ry(1.3440068912999719) q[3];
cx q[1],q[3];
ry(1.5838802007804607) q[3];
ry(2.238504579829688) q[5];
cx q[3],q[5];
ry(-0.3073178069042344) q[3];
ry(-0.8022679532492205) q[5];
cx q[3],q[5];
ry(-2.111954709659953) q[5];
ry(-0.021226313824312726) q[7];
cx q[5],q[7];
ry(3.121841592516951) q[5];
ry(-0.6999084483429705) q[7];
cx q[5],q[7];
ry(1.0988589846451025) q[0];
ry(1.9720342076764696) q[1];
cx q[0],q[1];
ry(1.4786870636240759) q[0];
ry(2.776227294605048) q[1];
cx q[0],q[1];
ry(0.2775361301525434) q[2];
ry(-0.10402613231089308) q[3];
cx q[2],q[3];
ry(2.949916053363607) q[2];
ry(1.7677155375151292) q[3];
cx q[2],q[3];
ry(0.6423067116024441) q[4];
ry(1.8432277006636102) q[5];
cx q[4],q[5];
ry(-0.06441947493533107) q[4];
ry(-2.476772970302153) q[5];
cx q[4],q[5];
ry(-0.7548505194373849) q[6];
ry(-0.6438122794583734) q[7];
cx q[6],q[7];
ry(0.045361828963624805) q[6];
ry(-1.347844018034863) q[7];
cx q[6],q[7];
ry(-1.5914188769494233) q[0];
ry(2.3971706591746593) q[2];
cx q[0],q[2];
ry(-2.3320294909726607) q[0];
ry(-1.5865948718474803) q[2];
cx q[0],q[2];
ry(2.1958294748751044) q[2];
ry(-0.15568670414884522) q[4];
cx q[2],q[4];
ry(-0.930787808989523) q[2];
ry(-1.049188919593358) q[4];
cx q[2],q[4];
ry(2.8612674836547747) q[4];
ry(-2.6310887847756064) q[6];
cx q[4],q[6];
ry(1.7523641530923548) q[4];
ry(-0.09077085795086225) q[6];
cx q[4],q[6];
ry(0.9970137849601267) q[1];
ry(-0.14148240273746016) q[3];
cx q[1],q[3];
ry(-2.474304254760259) q[1];
ry(-2.2910151006597004) q[3];
cx q[1],q[3];
ry(-3.0123932623909404) q[3];
ry(-1.5277500401915889) q[5];
cx q[3],q[5];
ry(-0.813606493508435) q[3];
ry(1.969648901267667) q[5];
cx q[3],q[5];
ry(0.5637028332423224) q[5];
ry(-2.541970043613224) q[7];
cx q[5],q[7];
ry(-2.305419847434468) q[5];
ry(-1.757871584094696) q[7];
cx q[5],q[7];
ry(-0.49958104831637884) q[0];
ry(-2.508836004052717) q[1];
cx q[0],q[1];
ry(-1.2229574012491886) q[0];
ry(0.7655589475802292) q[1];
cx q[0],q[1];
ry(0.297127902773215) q[2];
ry(0.35631377611123605) q[3];
cx q[2],q[3];
ry(-2.2757633086555105) q[2];
ry(-3.0335304550658475) q[3];
cx q[2],q[3];
ry(0.36311279346590797) q[4];
ry(-1.0074666344595782) q[5];
cx q[4],q[5];
ry(-1.0825358555378648) q[4];
ry(-1.7230833330501287) q[5];
cx q[4],q[5];
ry(2.0978510827443024) q[6];
ry(0.9210434494189608) q[7];
cx q[6],q[7];
ry(-0.6990571912301338) q[6];
ry(0.7732494793580553) q[7];
cx q[6],q[7];
ry(1.0502919865985532) q[0];
ry(-0.8396818829452991) q[2];
cx q[0],q[2];
ry(-2.2382861841505513) q[0];
ry(0.8363717892457574) q[2];
cx q[0],q[2];
ry(1.4747846318486966) q[2];
ry(-1.585340046892836) q[4];
cx q[2],q[4];
ry(-0.7706204117346991) q[2];
ry(2.795177618667758) q[4];
cx q[2],q[4];
ry(-2.8287590642545717) q[4];
ry(2.5236356387898953) q[6];
cx q[4],q[6];
ry(-2.1030615770981744) q[4];
ry(-2.5506399939988555) q[6];
cx q[4],q[6];
ry(-1.8838693755507336) q[1];
ry(-0.02123927629536048) q[3];
cx q[1],q[3];
ry(-2.3597782273403545) q[1];
ry(1.9938878788234202) q[3];
cx q[1],q[3];
ry(-2.6288191819034648) q[3];
ry(-2.1481858213950056) q[5];
cx q[3],q[5];
ry(-0.8491882559603003) q[3];
ry(2.4947812231866466) q[5];
cx q[3],q[5];
ry(0.8683618545804882) q[5];
ry(1.1762477667273983) q[7];
cx q[5],q[7];
ry(-0.5892332189786247) q[5];
ry(1.99573053514664) q[7];
cx q[5],q[7];
ry(2.431114611717832) q[0];
ry(-2.1326654632696784) q[1];
cx q[0],q[1];
ry(0.5536947160609804) q[0];
ry(0.02420162278947453) q[1];
cx q[0],q[1];
ry(2.9870000661962606) q[2];
ry(-2.906753689841066) q[3];
cx q[2],q[3];
ry(-2.9721238939522268) q[2];
ry(-0.7099695732086113) q[3];
cx q[2],q[3];
ry(2.1723291543554404) q[4];
ry(1.915740381614465) q[5];
cx q[4],q[5];
ry(0.13835307538626207) q[4];
ry(-0.972111134728527) q[5];
cx q[4],q[5];
ry(2.3136191683810696) q[6];
ry(-1.644604846130521) q[7];
cx q[6],q[7];
ry(1.8711801045641419) q[6];
ry(-1.109927942706081) q[7];
cx q[6],q[7];
ry(1.737455369671134) q[0];
ry(-0.41769970166042114) q[2];
cx q[0],q[2];
ry(-1.4245417183222104) q[0];
ry(0.20173755081496034) q[2];
cx q[0],q[2];
ry(2.5733357320544896) q[2];
ry(2.2859225510839614) q[4];
cx q[2],q[4];
ry(-0.24822831440771154) q[2];
ry(-0.7456034750279859) q[4];
cx q[2],q[4];
ry(-1.6008141596166061) q[4];
ry(-2.759505035584373) q[6];
cx q[4],q[6];
ry(-0.6833731973413188) q[4];
ry(1.3696228093391676) q[6];
cx q[4],q[6];
ry(-1.0263646217683384) q[1];
ry(0.1229830063778848) q[3];
cx q[1],q[3];
ry(0.29940539962159995) q[1];
ry(-2.7254499512986974) q[3];
cx q[1],q[3];
ry(-1.2362408237438223) q[3];
ry(-0.8427854922747928) q[5];
cx q[3],q[5];
ry(2.875144636747873) q[3];
ry(1.4591780073468152) q[5];
cx q[3],q[5];
ry(-0.056129129184387616) q[5];
ry(0.14996808332463876) q[7];
cx q[5],q[7];
ry(-0.21608642470704886) q[5];
ry(-1.536655588920711) q[7];
cx q[5],q[7];
ry(1.7155594755034822) q[0];
ry(-1.9071776434154817) q[1];
cx q[0],q[1];
ry(-2.8639851016788653) q[0];
ry(0.18381211672040945) q[1];
cx q[0],q[1];
ry(2.563341414035951) q[2];
ry(-2.506777890707293) q[3];
cx q[2],q[3];
ry(1.2210929238417707) q[2];
ry(1.2244798275319928) q[3];
cx q[2],q[3];
ry(1.7626022125577494) q[4];
ry(-0.002391482151630534) q[5];
cx q[4],q[5];
ry(1.2764354524736468) q[4];
ry(-1.957678103093973) q[5];
cx q[4],q[5];
ry(-1.9888573997573396) q[6];
ry(-2.1352152310297097) q[7];
cx q[6],q[7];
ry(3.0079456858886493) q[6];
ry(2.361277424116915) q[7];
cx q[6],q[7];
ry(-2.4951099372864847) q[0];
ry(1.656783557568957) q[2];
cx q[0],q[2];
ry(-1.6432388222483452) q[0];
ry(2.9409603824512027) q[2];
cx q[0],q[2];
ry(1.573766954531935) q[2];
ry(-1.7639452924092573) q[4];
cx q[2],q[4];
ry(-2.2391537124195278) q[2];
ry(-2.5034842560300095) q[4];
cx q[2],q[4];
ry(-2.017512465932911) q[4];
ry(-2.9638227532197217) q[6];
cx q[4],q[6];
ry(-2.2474725226912806) q[4];
ry(-0.5342797613339378) q[6];
cx q[4],q[6];
ry(0.10156186812514717) q[1];
ry(3.0581692598770953) q[3];
cx q[1],q[3];
ry(-1.1287599965704018) q[1];
ry(-0.23547565871001844) q[3];
cx q[1],q[3];
ry(0.8135046479734256) q[3];
ry(1.8250350134096125) q[5];
cx q[3],q[5];
ry(-1.278628377418935) q[3];
ry(-0.8377947837706657) q[5];
cx q[3],q[5];
ry(-2.1303481869013385) q[5];
ry(1.2499438704190862) q[7];
cx q[5],q[7];
ry(2.28789533738325) q[5];
ry(-1.7074569824072248) q[7];
cx q[5],q[7];
ry(-0.5764562640806332) q[0];
ry(3.1255414152257965) q[1];
cx q[0],q[1];
ry(-0.6035237313326474) q[0];
ry(-2.529412175259105) q[1];
cx q[0],q[1];
ry(-1.0315218412949951) q[2];
ry(-1.2592316962640773) q[3];
cx q[2],q[3];
ry(0.8185981779590623) q[2];
ry(-3.1143208084226996) q[3];
cx q[2],q[3];
ry(0.11629636612743446) q[4];
ry(1.4092162394621761) q[5];
cx q[4],q[5];
ry(3.0486985454935454) q[4];
ry(-2.6909124314601027) q[5];
cx q[4],q[5];
ry(-2.859664223966614) q[6];
ry(2.0973616601488736) q[7];
cx q[6],q[7];
ry(-2.511247048824253) q[6];
ry(2.2134820826302306) q[7];
cx q[6],q[7];
ry(-0.7404211628104811) q[0];
ry(1.868941079904399) q[2];
cx q[0],q[2];
ry(1.2670658605065386) q[0];
ry(0.636163191829735) q[2];
cx q[0],q[2];
ry(-1.7191037895086145) q[2];
ry(3.0251425445662887) q[4];
cx q[2],q[4];
ry(-2.611712336597922) q[2];
ry(2.5218457824601654) q[4];
cx q[2],q[4];
ry(-2.1971934793143677) q[4];
ry(1.7115740327451316) q[6];
cx q[4],q[6];
ry(-1.4136424592818875) q[4];
ry(3.0113445376243444) q[6];
cx q[4],q[6];
ry(-2.4272606720405694) q[1];
ry(1.6827611212950129) q[3];
cx q[1],q[3];
ry(1.1887532364947295) q[1];
ry(-0.29726315070466686) q[3];
cx q[1],q[3];
ry(-2.816010585635201) q[3];
ry(2.7333630825727226) q[5];
cx q[3],q[5];
ry(-2.717793001494509) q[3];
ry(-1.6339383118301627) q[5];
cx q[3],q[5];
ry(1.0184555889584448) q[5];
ry(1.343668837287125) q[7];
cx q[5],q[7];
ry(-0.9977193927642769) q[5];
ry(1.1258588060784698) q[7];
cx q[5],q[7];
ry(0.5158517288464122) q[0];
ry(2.6559528898451057) q[1];
cx q[0],q[1];
ry(-2.721568913042523) q[0];
ry(-1.1164280351601592) q[1];
cx q[0],q[1];
ry(0.7562314388354854) q[2];
ry(0.551345688120745) q[3];
cx q[2],q[3];
ry(-0.22396715349387425) q[2];
ry(2.0177327834893433) q[3];
cx q[2],q[3];
ry(-2.6760138956548776) q[4];
ry(0.58239664673768) q[5];
cx q[4],q[5];
ry(1.6453656000855228) q[4];
ry(1.6916622697895634) q[5];
cx q[4],q[5];
ry(1.4693884662064105) q[6];
ry(3.091631074100697) q[7];
cx q[6],q[7];
ry(2.0848144068754917) q[6];
ry(-2.6203791876379166) q[7];
cx q[6],q[7];
ry(2.196558294898585) q[0];
ry(-0.8759640581781394) q[2];
cx q[0],q[2];
ry(0.8015749814132399) q[0];
ry(2.6939909297212092) q[2];
cx q[0],q[2];
ry(1.0267829108834823) q[2];
ry(-1.6616817268472293) q[4];
cx q[2],q[4];
ry(2.403897210827891) q[2];
ry(-1.6675687316400039) q[4];
cx q[2],q[4];
ry(-0.9272256393565108) q[4];
ry(-0.07702473523821353) q[6];
cx q[4],q[6];
ry(0.7476238436932229) q[4];
ry(-0.599448161439601) q[6];
cx q[4],q[6];
ry(2.7786568076171427) q[1];
ry(3.064082245233417) q[3];
cx q[1],q[3];
ry(-2.844895146075469) q[1];
ry(1.9921235992856823) q[3];
cx q[1],q[3];
ry(0.3651260735779287) q[3];
ry(-2.3737035542315397) q[5];
cx q[3],q[5];
ry(3.014531733252844) q[3];
ry(2.0672754377787017) q[5];
cx q[3],q[5];
ry(-3.129953020522472) q[5];
ry(2.7993004960228283) q[7];
cx q[5],q[7];
ry(-0.7460088685117858) q[5];
ry(-0.3066460004024085) q[7];
cx q[5],q[7];
ry(1.7865132887281792) q[0];
ry(0.13777320773210278) q[1];
cx q[0],q[1];
ry(2.3425721047076618) q[0];
ry(-2.149681786854714) q[1];
cx q[0],q[1];
ry(-0.9643226634410063) q[2];
ry(-1.199187175394722) q[3];
cx q[2],q[3];
ry(-1.4745040994429104) q[2];
ry(-2.3600547566620707) q[3];
cx q[2],q[3];
ry(0.27578395629088437) q[4];
ry(3.135712259608002) q[5];
cx q[4],q[5];
ry(-1.412228303830946) q[4];
ry(-1.0212283981932853) q[5];
cx q[4],q[5];
ry(-1.4782567745025645) q[6];
ry(2.0305605740172554) q[7];
cx q[6],q[7];
ry(-1.8039828817260088) q[6];
ry(2.0306713453510032) q[7];
cx q[6],q[7];
ry(1.8952730189276585) q[0];
ry(0.8453175720183781) q[2];
cx q[0],q[2];
ry(-1.9329480315404706) q[0];
ry(1.8701358196389555) q[2];
cx q[0],q[2];
ry(-1.4632134747685364) q[2];
ry(-1.13389586694358) q[4];
cx q[2],q[4];
ry(0.9108191349339876) q[2];
ry(-2.9661837091043703) q[4];
cx q[2],q[4];
ry(2.5196892032272413) q[4];
ry(-1.285736230055896) q[6];
cx q[4],q[6];
ry(-1.1139642994805625) q[4];
ry(2.0293311561090817) q[6];
cx q[4],q[6];
ry(-1.9601836688656347) q[1];
ry(-3.131558379638035) q[3];
cx q[1],q[3];
ry(-0.13251802471754637) q[1];
ry(-0.41387626112881826) q[3];
cx q[1],q[3];
ry(1.7101444741528986) q[3];
ry(-2.3194113204720637) q[5];
cx q[3],q[5];
ry(-1.7965201922089156) q[3];
ry(-0.24016653433629964) q[5];
cx q[3],q[5];
ry(2.7761804965775387) q[5];
ry(2.4101872677562404) q[7];
cx q[5],q[7];
ry(-0.9787883998655847) q[5];
ry(1.8737220779865327) q[7];
cx q[5],q[7];
ry(-1.1711540993663299) q[0];
ry(0.9138528021482104) q[1];
cx q[0],q[1];
ry(-2.037316854811642) q[0];
ry(0.49562591428702374) q[1];
cx q[0],q[1];
ry(2.4276867661068495) q[2];
ry(-0.10369965946445897) q[3];
cx q[2],q[3];
ry(-1.918156466413035) q[2];
ry(1.6715395846606067) q[3];
cx q[2],q[3];
ry(-0.49850473584095756) q[4];
ry(2.866021755225726) q[5];
cx q[4],q[5];
ry(1.5012428760723908) q[4];
ry(2.6302923487471888) q[5];
cx q[4],q[5];
ry(-3.0789924783690674) q[6];
ry(1.3915684656082465) q[7];
cx q[6],q[7];
ry(-1.763624837914243) q[6];
ry(1.3849641139631346) q[7];
cx q[6],q[7];
ry(-0.5857574404926593) q[0];
ry(0.39757694724507964) q[2];
cx q[0],q[2];
ry(-0.12381969788100111) q[0];
ry(-0.8576168088199037) q[2];
cx q[0],q[2];
ry(-0.57989638601377) q[2];
ry(1.2486584175691657) q[4];
cx q[2],q[4];
ry(-3.0263082320809933) q[2];
ry(0.11240278609608415) q[4];
cx q[2],q[4];
ry(-0.7704852963420148) q[4];
ry(-1.8584343600338338) q[6];
cx q[4],q[6];
ry(-2.962014803673383) q[4];
ry(-0.28973894753410706) q[6];
cx q[4],q[6];
ry(-1.7492380867053434) q[1];
ry(-0.18586158397925148) q[3];
cx q[1],q[3];
ry(1.8208440320645085) q[1];
ry(2.446132858223213) q[3];
cx q[1],q[3];
ry(0.6436829655377723) q[3];
ry(-1.5063869474998) q[5];
cx q[3],q[5];
ry(1.1518056705394697) q[3];
ry(-0.2272084756663828) q[5];
cx q[3],q[5];
ry(0.22997777096185842) q[5];
ry(-0.3326371801570076) q[7];
cx q[5],q[7];
ry(-2.508308994059527) q[5];
ry(-2.4486575735704843) q[7];
cx q[5],q[7];
ry(3.0955215855473543) q[0];
ry(-0.8840740918347159) q[1];
cx q[0],q[1];
ry(-0.41177305785680396) q[0];
ry(0.8969071239059548) q[1];
cx q[0],q[1];
ry(-0.26042473330439425) q[2];
ry(-2.139099520507604) q[3];
cx q[2],q[3];
ry(0.1867511915864286) q[2];
ry(0.2608857950059438) q[3];
cx q[2],q[3];
ry(0.8589799689303893) q[4];
ry(1.5330105608697657) q[5];
cx q[4],q[5];
ry(1.2053883108487853) q[4];
ry(0.45885292686742574) q[5];
cx q[4],q[5];
ry(-1.1154470853133054) q[6];
ry(-0.3056689039175611) q[7];
cx q[6],q[7];
ry(2.8874306167531105) q[6];
ry(-0.6122913092381201) q[7];
cx q[6],q[7];
ry(-0.21294217627452472) q[0];
ry(-2.22948519563266) q[2];
cx q[0],q[2];
ry(0.1885850173928007) q[0];
ry(2.3160581284904915) q[2];
cx q[0],q[2];
ry(2.6205435328040854) q[2];
ry(1.1573703945381295) q[4];
cx q[2],q[4];
ry(-2.0434127952334746) q[2];
ry(1.5126654242592614) q[4];
cx q[2],q[4];
ry(1.0965751343033538) q[4];
ry(0.5089263953300787) q[6];
cx q[4],q[6];
ry(-1.1001983731316738) q[4];
ry(-1.14436268553433) q[6];
cx q[4],q[6];
ry(-2.8524794435181744) q[1];
ry(-1.1704121743906466) q[3];
cx q[1],q[3];
ry(-0.2870022473032776) q[1];
ry(-0.3613652437541069) q[3];
cx q[1],q[3];
ry(-2.376649404071717) q[3];
ry(-1.7103054734039702) q[5];
cx q[3],q[5];
ry(0.8531705101994438) q[3];
ry(-2.5972531279552005) q[5];
cx q[3],q[5];
ry(-1.7727841321847508) q[5];
ry(2.181668058193969) q[7];
cx q[5],q[7];
ry(2.8920834508796585) q[5];
ry(-0.16567706291328868) q[7];
cx q[5],q[7];
ry(-2.92103660014129) q[0];
ry(1.8619394497896382) q[1];
cx q[0],q[1];
ry(0.257127240682256) q[0];
ry(-2.0842330657110795) q[1];
cx q[0],q[1];
ry(2.08908889434875) q[2];
ry(1.9361152899620784) q[3];
cx q[2],q[3];
ry(2.1748984613322655) q[2];
ry(-0.38921972141507605) q[3];
cx q[2],q[3];
ry(0.8135107321660505) q[4];
ry(-0.14231942491022617) q[5];
cx q[4],q[5];
ry(-0.43051865245585663) q[4];
ry(-1.524817800075997) q[5];
cx q[4],q[5];
ry(-1.2922184272098685) q[6];
ry(2.42057668719066) q[7];
cx q[6],q[7];
ry(-0.4528878014505855) q[6];
ry(1.880859084342228) q[7];
cx q[6],q[7];
ry(2.696535034364022) q[0];
ry(-1.7987279693968843) q[2];
cx q[0],q[2];
ry(2.3233405204092468) q[0];
ry(2.857658976700002) q[2];
cx q[0],q[2];
ry(-0.08685312707937384) q[2];
ry(2.605005864710649) q[4];
cx q[2],q[4];
ry(2.4170799056031584) q[2];
ry(-3.0798528152348132) q[4];
cx q[2],q[4];
ry(2.787624290121577) q[4];
ry(0.9445737933103979) q[6];
cx q[4],q[6];
ry(1.329219824354838) q[4];
ry(-0.5958631320067482) q[6];
cx q[4],q[6];
ry(-0.6807799299280504) q[1];
ry(-0.9525160670499997) q[3];
cx q[1],q[3];
ry(1.476562721682324) q[1];
ry(-1.9161452467760067) q[3];
cx q[1],q[3];
ry(1.5122039204776208) q[3];
ry(1.1229276109002937) q[5];
cx q[3],q[5];
ry(0.8862551110218152) q[3];
ry(-0.6217476198534879) q[5];
cx q[3],q[5];
ry(2.379386373827497) q[5];
ry(2.7843356716485235) q[7];
cx q[5],q[7];
ry(2.702822949141308) q[5];
ry(2.2688976569093118) q[7];
cx q[5],q[7];
ry(0.8694811627370562) q[0];
ry(-1.8551651295478238) q[1];
cx q[0],q[1];
ry(0.6855981952603231) q[0];
ry(-0.21930772381679553) q[1];
cx q[0],q[1];
ry(1.7000168918013925) q[2];
ry(0.16644378052364406) q[3];
cx q[2],q[3];
ry(-0.7753619898626668) q[2];
ry(-1.4679950508664494) q[3];
cx q[2],q[3];
ry(2.0857558450236393) q[4];
ry(-2.719459408003923) q[5];
cx q[4],q[5];
ry(-3.0533065780972293) q[4];
ry(0.1959064862702915) q[5];
cx q[4],q[5];
ry(2.3613600191142097) q[6];
ry(-2.8178638920725367) q[7];
cx q[6],q[7];
ry(2.9223839562126126) q[6];
ry(0.34692450602018665) q[7];
cx q[6],q[7];
ry(0.3992034235930477) q[0];
ry(1.3276236779927129) q[2];
cx q[0],q[2];
ry(0.6574429414286636) q[0];
ry(-1.4966893794011147) q[2];
cx q[0],q[2];
ry(1.5987115386767787) q[2];
ry(0.6036929818286637) q[4];
cx q[2],q[4];
ry(2.9546506715249135) q[2];
ry(2.202233850560545) q[4];
cx q[2],q[4];
ry(-0.35633922338666757) q[4];
ry(-2.9667255305917237) q[6];
cx q[4],q[6];
ry(-0.4486220780440977) q[4];
ry(0.16297233164247338) q[6];
cx q[4],q[6];
ry(-0.13596986412232948) q[1];
ry(-0.4007287477010033) q[3];
cx q[1],q[3];
ry(1.823478790421154) q[1];
ry(-2.2363847261813277) q[3];
cx q[1],q[3];
ry(1.3877054574197176) q[3];
ry(0.08501675780615958) q[5];
cx q[3],q[5];
ry(0.9614196668927198) q[3];
ry(2.9586525320374224) q[5];
cx q[3],q[5];
ry(2.8643209920662964) q[5];
ry(1.070546327962055) q[7];
cx q[5],q[7];
ry(-0.5334004593844339) q[5];
ry(0.21886650478680014) q[7];
cx q[5],q[7];
ry(-0.32668837343242707) q[0];
ry(-1.3797528307686981) q[1];
ry(-2.8749127979143028) q[2];
ry(0.43638998376229354) q[3];
ry(-1.1844285831081072) q[4];
ry(-0.7651947153280991) q[5];
ry(-1.7062070331905193) q[6];
ry(2.4286778044559183) q[7];