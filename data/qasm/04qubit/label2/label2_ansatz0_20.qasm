OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11274920737815777) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.02252305072103603) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04948075562908117) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20074396268435696) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.17221547820770552) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.041829536714817636) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.023171292487748393) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.052024271296654184) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05954815641110577) q[3];
cx q[2],q[3];
rx(0.42172821020188894) q[0];
rz(-0.07490689606130295) q[0];
rx(-0.1652524518326351) q[1];
rz(0.038992858967420246) q[1];
rx(-0.6630512758192418) q[2];
rz(-0.049053710126050316) q[2];
rx(0.23437925325258693) q[3];
rz(-0.07824266473948664) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1215889018653866) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.008415763319474297) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0729093046304241) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2968274629014105) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2348528940814252) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.028773555370291092) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10679709318158533) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1655696378993024) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09020789482964979) q[3];
cx q[2],q[3];
rx(0.4249565982866213) q[0];
rz(-0.07011395684089126) q[0];
rx(-0.24483624772253476) q[1];
rz(-0.005261927578534505) q[1];
rx(-0.6326201800644291) q[2];
rz(-0.20937474867818057) q[2];
rx(0.13127492071737323) q[3];
rz(-0.03742330053793143) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09530911201261738) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04914680351716878) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11679711507953856) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.301920625143964) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2713893281120925) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.026238986195652632) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05070669543904429) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.026060246362683982) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.18990930625556346) q[3];
cx q[2],q[3];
rx(0.4139208245530985) q[0];
rz(-0.1595133425161769) q[0];
rx(-0.29980392839240555) q[1];
rz(-0.05534441293721347) q[1];
rx(-0.48408105150155256) q[2];
rz(-0.32850456419531715) q[2];
rx(0.16592607466404077) q[3];
rz(0.002146705537412824) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.18597762237433355) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06408051416481483) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.18926584611642353) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23379426965490002) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.36612782920026044) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11145390212678051) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09937759993597381) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.13473632898929003) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.22937147591461224) q[3];
cx q[2],q[3];
rx(0.4532804777200717) q[0];
rz(-0.1989423890581013) q[0];
rx(-0.35175581479053625) q[1];
rz(-0.05191921794453337) q[1];
rx(-0.27658031288064766) q[2];
rz(-0.32652077652474204) q[2];
rx(0.0998879762830212) q[3];
rz(0.04992235362261427) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.21419776699065218) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17413275802678124) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09698728018176438) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.23985135278136363) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.42379518842665714) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10609304187511549) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06998485278537851) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3296799113547899) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2827466241178726) q[3];
cx q[2],q[3];
rx(0.3744005053027389) q[0];
rz(-0.10004785744913312) q[0];
rx(-0.41415245475754675) q[1];
rz(-0.08617693734433106) q[1];
rx(-0.12486981684160842) q[2];
rz(-0.2738447581004231) q[2];
rx(0.017051406830066732) q[3];
rz(0.0865398961235602) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.19808517099988404) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1477291182761282) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.052526036827549555) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.213682918583644) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2657920008752796) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.004709646826928418) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.19112224954067547) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3875626644960514) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15079626944460015) q[3];
cx q[2],q[3];
rx(0.3666609243433059) q[0];
rz(-0.02699289596281305) q[0];
rx(-0.49271246751744313) q[1];
rz(-0.11476327692462239) q[1];
rx(-0.09563401968405179) q[2];
rz(-0.15347463454034896) q[2];
rx(0.008434894466138404) q[3];
rz(0.1448462578574891) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15757988243979185) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1737446149339611) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06324453461426063) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2670667416457669) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.034306426624226824) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06051116461380425) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1848390905987709) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.30636020796556374) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1240355645904137) q[3];
cx q[2],q[3];
rx(0.3972509814681482) q[0];
rz(0.08679665714745377) q[0];
rx(-0.5219751125355057) q[1];
rz(-0.10439965874814053) q[1];
rx(-0.0877223131336804) q[2];
rz(-0.10126360600984298) q[2];
rx(-0.09808243051613788) q[3];
rz(0.2521855338144998) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13968675126077637) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1339556242216787) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0555195173666419) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2648406393584283) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04396409687019494) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06766198899200274) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16122028701992352) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15822045920841663) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08012881856522895) q[3];
cx q[2],q[3];
rx(0.3568209258563733) q[0];
rz(0.11757760299045167) q[0];
rx(-0.549473551973291) q[1];
rz(-0.11955117781451902) q[1];
rx(-0.016768783850321184) q[2];
rz(-0.08911180381947234) q[2];
rx(-0.18019232005064528) q[3];
rz(0.26422652353546955) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07519039763713192) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1090761512184815) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04598896899996493) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2585834851139545) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.005723284486083958) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.01950915887351702) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0986986814445846) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.13487186903271806) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04449417701992005) q[3];
cx q[2],q[3];
rx(0.303242128123271) q[0];
rz(0.18701839686980226) q[0];
rx(-0.5087547469659774) q[1];
rz(-0.10557443536187106) q[1];
rx(0.05520598937969966) q[2];
rz(-0.1254430786081934) q[2];
rx(-0.26868209874939075) q[3];
rz(0.32274222727664587) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07075587326681763) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09383677825793821) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08696413210088334) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2170913077215992) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.040403636350585685) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.01954696857449936) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08708100649904182) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.023829986472480927) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04099914031782394) q[3];
cx q[2],q[3];
rx(0.25487484321310677) q[0];
rz(0.18477924523909267) q[0];
rx(-0.48288771499066846) q[1];
rz(-0.07413651190482776) q[1];
rx(0.019445099402660315) q[2];
rz(-0.16774414500981316) q[2];
rx(-0.24796888032200964) q[3];
rz(0.289413363686548) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09609346129200362) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.034784312081374884) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12476957651749189) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.013154653929374599) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11003975365195261) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06735905235994363) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04093973365503144) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17132604939911913) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0020452666095066327) q[3];
cx q[2],q[3];
rx(0.20782486274147627) q[0];
rz(0.2488935803479584) q[0];
rx(-0.47593308705013476) q[1];
rz(-0.09434716134393599) q[1];
rx(0.12786846782128483) q[2];
rz(-0.12534723553742405) q[2];
rx(-0.3034720931525196) q[3];
rz(0.3525469734559726) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12515850088433011) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05790900944004244) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.17908175266363707) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1446108356616191) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03225263583404343) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15068727282870187) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0002380275174420712) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2212349706617203) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.023685069958376288) q[3];
cx q[2],q[3];
rx(0.2109250765672955) q[0];
rz(0.31219193795089095) q[0];
rx(-0.37146885305749683) q[1];
rz(-0.06996846849345595) q[1];
rx(0.11802077335488073) q[2];
rz(-0.20785225890536124) q[2];
rx(-0.315519067613558) q[3];
rz(0.29263715433007004) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.01127358337634575) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08857283244142077) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.2712463986406982) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.3334496732008113) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.18492017913518274) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.17782366923645765) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.023025985503041555) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.22943636855699887) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07924109541578625) q[3];
cx q[2],q[3];
rx(0.15892841882816874) q[0];
rz(0.2663148346098514) q[0];
rx(-0.2521615729652386) q[1];
rz(-0.015540195766871524) q[1];
rx(0.047075714598095945) q[2];
rz(-0.2582407232354459) q[2];
rx(-0.4178284646158987) q[3];
rz(0.19065004240304914) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03239772677849912) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.10111090786630775) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.46556164963673674) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.24431543130983197) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.1772573631595208) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.28746851131842344) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12634319488655435) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.33799858096478747) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07721711271642867) q[3];
cx q[2],q[3];
rx(0.05131953072339398) q[0];
rz(0.2660642458227611) q[0];
rx(-0.08738766664795401) q[1];
rz(-0.08100068564111147) q[1];
rx(0.06213358370097154) q[2];
rz(-0.27195048253898885) q[2];
rx(-0.417374091495918) q[3];
rz(0.14852911315306833) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10804321792135727) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05824995996366371) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.4626568983698089) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1691074389970649) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.017043675403911527) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.29546662513129934) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11502238962363125) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3682299777826707) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.016112860330429832) q[3];
cx q[2],q[3];
rx(-0.058952586338113046) q[0];
rz(0.2343039479164474) q[0];
rx(0.033224792565183105) q[1];
rz(-0.15425782678232436) q[1];
rx(0.19564037345840005) q[2];
rz(-0.291169745711316) q[2];
rx(-0.35334336177622855) q[3];
rz(0.10408462089549078) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10324705375049857) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08086706914563395) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.39989229393264936) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.03013802825852639) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0025106990002109323) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1817278950767821) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12492603642973087) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2748781628501741) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.033302968650844415) q[3];
cx q[2],q[3];
rx(-0.17087649123013207) q[0];
rz(0.2396847281736546) q[0];
rx(0.10793572390528473) q[1];
rz(-0.2512348083277938) q[1];
rx(0.16064384570060755) q[2];
rz(-0.19867719790586386) q[2];
rx(-0.34435271908476944) q[3];
rz(0.11511454789973997) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07133747409887847) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.21552806383475265) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.19453820419752602) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.024245857436732274) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.003639407269177132) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1418412264776406) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09124767448560109) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.2959696523350763) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.018604884976835802) q[3];
cx q[2],q[3];
rx(-0.20912074014427953) q[0];
rz(0.24543113976158484) q[0];
rx(0.2516229523870778) q[1];
rz(-0.25537945207526247) q[1];
rx(0.22445396059204553) q[2];
rz(-0.23176269729047516) q[2];
rx(-0.34549331731822563) q[3];
rz(0.12116725207170675) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12685415357250462) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.26491313646201475) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12421420299057911) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09821036420156012) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.022791385372215266) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.003499848111069578) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10279817817076722) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.19645747092806456) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0699090683553297) q[3];
cx q[2],q[3];
rx(-0.22358558427495545) q[0];
rz(0.34170672302793953) q[0];
rx(0.2755490209187538) q[1];
rz(-0.348815332333507) q[1];
rx(0.18719483153026684) q[2];
rz(-0.17181019018098825) q[2];
rx(-0.2814512847462828) q[3];
rz(0.12169408661279515) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08354087822612409) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.21004302031471705) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05503233775203238) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14369955576859103) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.022403949992337326) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.015628405429173802) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07015924491978325) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1687009331731229) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0032908429025486864) q[3];
cx q[2],q[3];
rx(-0.19557621914794443) q[0];
rz(0.37522659360951055) q[0];
rx(0.3007961003566871) q[1];
rz(-0.45701666441238) q[1];
rx(0.11722563104874265) q[2];
rz(-0.05598726522526721) q[2];
rx(-0.26347502439992154) q[3];
rz(0.08418619234623426) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0306215068028893) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07518987705727007) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.040483367653765874) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2584620334807505) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0338434871765778) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07774833902172112) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1251758574474127) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.09686909342240473) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.021813678970670294) q[3];
cx q[2],q[3];
rx(-0.2865143132145396) q[0];
rz(0.30280091332363246) q[0];
rx(0.2768410420714734) q[1];
rz(-0.47709542809000144) q[1];
rx(0.10906578416162986) q[2];
rz(0.037694012752610515) q[2];
rx(-0.20325964090444845) q[3];
rz(-0.061479912550530154) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07418867073269074) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.21704679491468373) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13371241914398663) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.24582510765553556) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03140920955144522) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06632430014572883) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05452250486798307) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12288916832666508) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04030520313252895) q[3];
cx q[2],q[3];
rx(-0.36379391940238515) q[0];
rz(0.2345070067632918) q[0];
rx(0.23681002385131575) q[1];
rz(-0.4302250303183705) q[1];
rx(0.03597325693438471) q[2];
rz(0.22568289005501235) q[2];
rx(-0.19890657631040326) q[3];
rz(-0.20231175416952893) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.15197555922224462) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.3671909926475094) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09955081358834893) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10436169513538135) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.15174064642482413) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10087439129408221) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.012197917701082495) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04813682498102723) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03523944972959089) q[3];
cx q[2],q[3];
rx(-0.3286800586762741) q[0];
rz(0.1990187332074306) q[0];
rx(0.24822056293466613) q[1];
rz(-0.3790991312025957) q[1];
rx(0.09439545442425293) q[2];
rz(0.17972584346360373) q[2];
rx(-0.08277610676941301) q[3];
rz(-0.145846058424228) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.019232716513282896) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.23025978220931997) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12780634131181012) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.028274591392211848) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2003341307723725) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15168695977298585) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.013887037111749654) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.016679322813711494) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.017869365526226946) q[3];
cx q[2],q[3];
rx(-0.26520914012104063) q[0];
rz(0.03248876118591923) q[0];
rx(0.12868568129405622) q[1];
rz(-0.16062872618588978) q[1];
rx(0.050343964017227495) q[2];
rz(0.2270970939388628) q[2];
rx(-0.10264986613897835) q[3];
rz(-0.22273421046165345) q[3];