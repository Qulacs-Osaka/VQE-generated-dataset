OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.08517607294605872) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2161381850577452) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.018103184620824036) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.026618296783841085) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04129718906583379) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08540649483905599) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09493375185144892) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.018534911645051787) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.00886368585741372) q[3];
cx q[2],q[3];
rz(-0.11706225115645498) q[0];
rz(-0.027353048322390227) q[1];
rz(-0.08913040548529824) q[2];
rz(-0.13510396391642382) q[3];
rx(-0.05737933922503362) q[0];
rx(-0.2845774970399367) q[1];
rx(-0.12051470869314641) q[2];
rx(-0.3631770192053252) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.018856130803412755) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13754899054046885) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.013557367245427498) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10775868800140526) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03502837744057237) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10639099673289382) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08318688985025896) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.06501421721820903) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09759612735114896) q[3];
cx q[2],q[3];
rz(-0.09731198774931278) q[0];
rz(-0.05933290670359018) q[1];
rz(0.013478826389291695) q[2];
rz(-0.15295815688801792) q[3];
rx(-0.0820614807580138) q[0];
rx(-0.2760927190803655) q[1];
rx(-0.10826917194790078) q[2];
rx(-0.3117499639172966) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.021494314397585233) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.029377399828862606) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07579680365173197) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08491553444584381) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09942102259655897) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08553556674706092) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0066399410000663964) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.029719181072294217) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.15657208714389426) q[3];
cx q[2],q[3];
rz(-0.01735892717780043) q[0];
rz(-0.10445171654110566) q[1];
rz(0.05833395400361351) q[2];
rz(-0.2166372494573143) q[3];
rx(-0.12583057080705218) q[0];
rx(-0.19369414370272614) q[1];
rx(-0.22571376024140066) q[2];
rx(-0.26731773822669097) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.01969002286376736) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.002128699806057894) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.027535562133027967) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.14970280156005994) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.052573868350368626) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13761426084554296) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0394724179618029) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.048203910457572524) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.21505320236437042) q[3];
cx q[2],q[3];
rz(-0.028209936006016928) q[0];
rz(-0.0717251835165693) q[1];
rz(0.17227499157634504) q[2];
rz(-0.1835118741189922) q[3];
rx(-0.1804577774445424) q[0];
rx(-0.24544503630318285) q[1];
rx(-0.23018787565660392) q[2];
rx(-0.2821512179060522) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07783500243709426) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.025864335197042305) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.026365222120360136) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.05028893477229797) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.02397003545811487) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.14325482374013562) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08938555869775903) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07256972151189785) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.16337584761607105) q[3];
cx q[2],q[3];
rz(-0.025854070660070394) q[0];
rz(-0.03016388111156267) q[1];
rz(0.23403127376345464) q[2];
rz(-0.10336897647786875) q[3];
rx(-0.26790392269297786) q[0];
rx(-0.12063193729233489) q[1];
rx(-0.2919757031368591) q[2];
rx(-0.2352747240426789) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.02760623117426868) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.059159445767008385) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0430853830518933) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09115502807481188) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.110325222612701) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12947634859618945) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.044728290767007545) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.031809910887296004) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0047590219598363645) q[3];
cx q[2],q[3];
rz(0.030776476703186742) q[0];
rz(-0.004191750471066479) q[1];
rz(0.20098986291202584) q[2];
rz(-0.07722781821353686) q[3];
rx(-0.28875646279442724) q[0];
rx(-0.1390685191392635) q[1];
rx(-0.3151203359434155) q[2];
rx(-0.20243251046307412) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.013457867883975995) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.031701053234948054) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04250727297479831) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14910273865949522) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0713524806821353) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09812541532967371) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05306018587036276) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14288673300696936) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09433377118322438) q[3];
cx q[2],q[3];
rz(0.07865235634403303) q[0];
rz(-0.033564619204002555) q[1];
rz(0.20774951436242511) q[2];
rz(-0.03516135319120319) q[3];
rx(-0.22533376861861204) q[0];
rx(-0.09531148234549057) q[1];
rx(-0.3217536615544187) q[2];
rx(-0.22958856324241883) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.051714544031944865) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02552835029558207) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0015744008304188568) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18957726520864582) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07202037567195227) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.14786697959680709) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.045651675071662644) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1517063763850406) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.20452618747903537) q[3];
cx q[2],q[3];
rz(0.0057484237353159365) q[0];
rz(-0.038722382926737506) q[1];
rz(0.1636613712163422) q[2];
rz(0.022140667952004752) q[3];
rx(-0.29513229528570567) q[0];
rx(-0.1178864722216244) q[1];
rx(-0.4198119522435956) q[2];
rx(-0.17301469962786015) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.04792197807407911) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.06785265591115634) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0806996028898626) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.24107671015271986) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07348090428892302) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09628415856052931) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.012316471541429013) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.17600055365465936) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.28857434800500353) q[3];
cx q[2],q[3];
rz(0.04236577082249509) q[0];
rz(-0.04311146286245166) q[1];
rz(0.08181002113228408) q[2];
rz(0.06881715848377155) q[3];
rx(-0.2897977773342304) q[0];
rx(-0.08934430799158517) q[1];
rx(-0.4803645924238811) q[2];
rx(-0.20797892755375585) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02451967941934266) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.227377419456978) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.034944142611280364) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20045741494923278) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07077292858993074) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.16955409139388306) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0326781685320717) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.14079994492299017) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.22244528034684646) q[3];
cx q[2],q[3];
rz(-0.05610860268609687) q[0];
rz(-0.1257672213635616) q[1];
rz(0.07627982430844324) q[2];
rz(0.08374339071204356) q[3];
rx(-0.22576546369031125) q[0];
rx(-0.10730180984212447) q[1];
rx(-0.48309166771137074) q[2];
rx(-0.20330327731691927) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.01023902079073896) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.178707416356148) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02664171137447372) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09523798032083186) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.09188912327330068) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2956217790965138) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.042287255686342624) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.08837617602712415) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.005381947408622911) q[3];
cx q[2],q[3];
rz(-0.07344545707859369) q[0];
rz(-0.157945024815189) q[1];
rz(0.1391462556175228) q[2];
rz(0.08544909921994304) q[3];
rx(-0.21539507080630405) q[0];
rx(-0.07468443800171642) q[1];
rx(-0.46631569268814954) q[2];
rx(-0.24807925675844375) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.009237567893880647) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0651398956711814) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0963856302664576) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.047774233700047804) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0155127047333281) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.32854191829501284) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.10768819716744178) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03178571618654247) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1314884193721079) q[3];
cx q[2],q[3];
rz(-0.07212918304438458) q[0];
rz(-0.1411901260067401) q[1];
rz(0.14685266321082777) q[2];
rz(-0.004828446513840612) q[3];
rx(-0.2292885779607051) q[0];
rx(-0.004307566134820917) q[1];
rx(-0.42123895257078425) q[2];
rx(-0.15352918674384697) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.006435793513376008) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02082317295179589) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04226389027561201) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.03311615388828161) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.05142999627879395) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.28144965738962063) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.013881375775634867) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.08322060541740552) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.27541841854585575) q[3];
cx q[2],q[3];
rz(-0.02541408319679093) q[0];
rz(-0.2097098545667796) q[1];
rz(0.1294965461613892) q[2];
rz(-0.06736763545062385) q[3];
rx(-0.19211069061515484) q[0];
rx(-0.10264896537362102) q[1];
rx(-0.41989493902878083) q[2];
rx(-0.1940849395409172) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.005506686573295346) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07389170780498873) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.050359399836622076) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.02740879939767818) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.15680320705779577) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19854221283437065) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05777401874737871) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04489623213164356) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2235753986890526) q[3];
cx q[2],q[3];
rz(-0.11624236945249994) q[0];
rz(-0.14551044452442075) q[1];
rz(0.056215306746432145) q[2];
rz(-0.11768646750581062) q[3];
rx(-0.1621925506963961) q[0];
rx(-0.029321156009651368) q[1];
rx(-0.4047690175680596) q[2];
rx(-0.11173239301850217) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05099702264724229) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1149357724831558) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12437535330155301) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.03755096108354307) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.24827231696301064) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.21915953708734495) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07859595299943059) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03415729691673807) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.23360740637010022) q[3];
cx q[2],q[3];
rz(-0.08895703795488542) q[0];
rz(-0.1414978921285214) q[1];
rz(0.0008662075440020199) q[2];
rz(-0.11427792682948248) q[3];
rx(-0.11790576619659171) q[0];
rx(-0.10809554545961035) q[1];
rx(-0.4852965073499423) q[2];
rx(-0.07411681369549146) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.19756763452716342) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.025805815743729612) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.10523412829056947) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.03496035913589789) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.319571159938115) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06115124295401782) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08957111732634626) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.14904285713122675) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07408954968792564) q[3];
cx q[2],q[3];
rz(-0.18692428570910097) q[0];
rz(-0.0457109159044143) q[1];
rz(-0.023308467515432633) q[2];
rz(-0.12002205108155156) q[3];
rx(-0.10462237427804737) q[0];
rx(-0.05381243674931586) q[1];
rx(-0.45058657417390763) q[2];
rx(-0.081242316015488) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13947579618248337) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.010319424112020702) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06052276235566526) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.02336962351708773) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.40379947595002236) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10006184812369011) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05632279125232815) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0999095667677877) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0834406372730178) q[3];
cx q[2],q[3];
rz(-0.17682690963844003) q[0];
rz(-0.027335587060147985) q[1];
rz(0.08400590504524157) q[2];
rz(-0.25523879909828845) q[3];
rx(-0.04524594761597601) q[0];
rx(0.009014766172068728) q[1];
rx(-0.4331725919075848) q[2];
rx(0.006962154821789559) q[3];