OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.01875916840535425) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09154458713118094) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0953703415380039) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.04430352055322382) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09106419503308857) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06365111484697912) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.025221911003579157) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04018699652753039) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0029570286516613625) q[3];
cx q[2],q[3];
rx(-0.19306225213170172) q[0];
rz(-0.03065230960732015) q[0];
rx(-0.10974163332554733) q[1];
rz(-0.06149735760139004) q[1];
rx(0.08732672596122795) q[2];
rz(-0.13753819294723674) q[2];
rx(-0.14206774938265077) q[3];
rz(-0.05732889610172579) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.028969000272558183) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.10092307494612714) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0958778416271696) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.025602600986671012) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.04973048279642787) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.17001862243045437) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06293943953638086) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.00523177486840273) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0961471680876303) q[3];
cx q[2],q[3];
rx(-0.14417429152779046) q[0];
rz(-0.06161424223890018) q[0];
rx(-0.06355226485148646) q[1];
rz(-0.10443679184979103) q[1];
rx(0.23185580968931052) q[2];
rz(-0.06669812154688388) q[2];
rx(-0.17979836188977488) q[3];
rz(-0.04229535931406132) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07038833853164872) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11232521387784561) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09842957296108643) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1296864743164182) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.02098164719406149) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.21805859875572453) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06168162255374981) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.06184069857405249) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.018402428206162995) q[3];
cx q[2],q[3];
rx(-0.17851227623540003) q[0];
rz(-0.04106294792990419) q[0];
rx(-0.012481087345016014) q[1];
rz(-0.0972203784477874) q[1];
rx(0.2688829654577083) q[2];
rz(-0.0553960696773831) q[2];
rx(-0.245065554974161) q[3];
rz(-0.05620939692639333) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07030664549007482) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.16931280828823916) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02415101842891621) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.24763759019229867) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07839826571225735) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.316517801232111) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.03965131459425977) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11053621563637996) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04438550776275075) q[3];
cx q[2],q[3];
rx(-0.23959170443710884) q[0];
rz(-0.0291298700916537) q[0];
rx(0.01188203393448405) q[1];
rz(-0.01867422743409428) q[1];
rx(0.3374662850587853) q[2];
rz(-0.08600812511664184) q[2];
rx(-0.2756925141834478) q[3];
rz(0.010040431590470661) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05315513133238165) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05959631442042406) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07441010483377847) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.3234897575234461) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07824560219725342) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.31244077872214276) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.039277621389890424) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.15051930842013184) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.013513441279400296) q[3];
cx q[2],q[3];
rx(-0.17141925425259646) q[0];
rz(-0.01709757963063603) q[0];
rx(0.018266703225405162) q[1];
rz(0.07301162618196474) q[1];
rx(0.3840698793936614) q[2];
rz(-0.03952328342287411) q[2];
rx(-0.38236646640293365) q[3];
rz(-0.012837764557598243) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.027434579015167083) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05566103597876093) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.02373837096683881) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.35157410102187864) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.006147087922714138) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.2759228694945335) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04793717513343392) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.13541071443216351) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08708466319239505) q[3];
cx q[2],q[3];
rx(-0.23070450738267345) q[0];
rz(0.01202819155951082) q[0];
rx(-0.03428385684460791) q[1];
rz(0.10904131125168987) q[1];
rx(0.43352398384387025) q[2];
rz(-0.0947680129713162) q[2];
rx(-0.36676586869035266) q[3];
rz(-0.018242303221049163) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08013581022649109) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.009467268817785764) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04766140711021404) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.38959024112902335) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.06092095049853652) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19651342852759351) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09721254036995465) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12995826061415322) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.022284059703599177) q[3];
cx q[2],q[3];
rx(-0.1769218834611063) q[0];
rz(0.019545988049717872) q[0];
rx(0.015483931097658221) q[1];
rz(0.09111179229607609) q[1];
rx(0.3740018660064999) q[2];
rz(-0.043298075873472626) q[2];
rx(-0.37174011529176404) q[3];
rz(0.005710146517195181) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0644960883998652) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09002155047916552) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.033705425856033894) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.29631875784139466) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11757400675880901) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13934703056753897) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.032885263850552) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1594496400274581) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04183004441859326) q[3];
cx q[2],q[3];
rx(-0.20893643223225603) q[0];
rz(0.02928228092094113) q[0];
rx(-0.0447491610335801) q[1];
rz(0.12474718170261294) q[1];
rx(0.39000894026396804) q[2];
rz(-0.16015677315039706) q[2];
rx(-0.39021658195884606) q[3];
rz(0.03367668862519016) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06806529004057253) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07144182737961621) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.012670119215359436) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.14414628800160284) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1528262624142375) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13408314225757367) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11753516926036685) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12624090534709093) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.011143625529507047) q[3];
cx q[2],q[3];
rx(-0.2059435077268206) q[0];
rz(0.09250190540630496) q[0];
rx(-0.03807691794182398) q[1];
rz(0.13093461078112237) q[1];
rx(0.3585614512571157) q[2];
rz(-0.1289841318077252) q[2];
rx(-0.41431676893974895) q[3];
rz(0.026839271528785427) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06493320286403265) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08533670491713978) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0001293668863476346) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.014252301470843524) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.16819811561570414) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1056808653446219) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.16524537094834077) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.054666343410403474) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0384479071751928) q[3];
cx q[2],q[3];
rx(-0.21151023294656224) q[0];
rz(0.09859431014130501) q[0];
rx(-0.04145312759388781) q[1];
rz(0.0811540891047293) q[1];
rx(0.2954570929873705) q[2];
rz(-0.030509185091288398) q[2];
rx(-0.356491500972667) q[3];
rz(-0.022952932491732112) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08801605382772328) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.024490549248554237) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07132691481638904) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09135574715067094) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.23863390067559892) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1318998132038699) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1514255982888572) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.03349390342923197) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04321483443317144) q[3];
cx q[2],q[3];
rx(-0.20889858057636876) q[0];
rz(0.0691838912731808) q[0];
rx(0.025536614515216785) q[1];
rz(0.0997055644425203) q[1];
rx(0.2639772437906319) q[2];
rz(-0.01856481198330715) q[2];
rx(-0.32604881169858485) q[3];
rz(0.031861188256509944) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11555304203447808) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.07969111251669773) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0743711178128992) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.17738088628370313) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.22073257098148588) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.05285638510040434) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.10972023887744149) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.00362035793830491) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14669333009022345) q[3];
cx q[2],q[3];
rx(-0.15616352196082622) q[0];
rz(0.09263816259319405) q[0];
rx(0.0823774066075838) q[1];
rz(-0.02566264982365948) q[1];
rx(0.21802908325075704) q[2];
rz(-0.02311169155925703) q[2];
rx(-0.3442729292128618) q[3];
rz(0.015579758195383878) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03119279957661823) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.09726264704306188) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.02953994437838537) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.25951643976180877) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1352065510781082) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.048644598449333776) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1552736833640366) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.05486631674013086) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.18703080223597252) q[3];
cx q[2],q[3];
rx(-0.17137556018841457) q[0];
rz(0.049478102881501645) q[0];
rx(0.132013247233658) q[1];
rz(-0.13000198752451167) q[1];
rx(0.18863996958109147) q[2];
rz(0.05628327241708024) q[2];
rx(-0.27942377113735156) q[3];
rz(0.007000270260624195) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08599835189543052) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.09905313434818348) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.10624890140285097) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2791156510184563) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.19232258352812417) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06542707170992737) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09909256116015117) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.012205095986167093) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11152217741640941) q[3];
cx q[2],q[3];
rx(-0.19936802200951426) q[0];
rz(-0.03759200517249189) q[0];
rx(0.09281852872332377) q[1];
rz(-0.26640076993829764) q[1];
rx(0.05902775667128041) q[2];
rz(0.1403618665833505) q[2];
rx(-0.19964944002473062) q[3];
rz(0.027947280238040512) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1081478348915937) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05347429070910196) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.17327929305263592) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3688404528410903) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24771433461923553) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.009045780961323675) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05863339512837053) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.024157881381497374) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11365091152947807) q[3];
cx q[2],q[3];
rx(-0.1740158814845983) q[0];
rz(-0.0329177885303056) q[0];
rx(0.03611804706744186) q[1];
rz(-0.26037492309182186) q[1];
rx(-0.012639567398444526) q[2];
rz(0.11922167498168197) q[2];
rx(-0.12206116101416116) q[3];
rz(-0.07871676892293328) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06212173871512464) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.13217650775975967) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1087074784380758) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3594135389944165) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1241744071399881) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.027189476927801968) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.032155592882465014) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.012787224325381452) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13030810354323952) q[3];
cx q[2],q[3];
rx(-0.21034933847942985) q[0];
rz(-0.017207192888447888) q[0];
rx(0.02713494644452135) q[1];
rz(-0.3114065845343137) q[1];
rx(-0.09830592872511756) q[2];
rz(0.18830141091108663) q[2];
rx(-0.03228901530872455) q[3];
rz(-0.023521643584788737) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0971777694675323) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.12893844951079766) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.00458303338569734) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.43170607949409096) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.12030843316592787) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.09201212905726419) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.012761611457855253) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.05769816709835256) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06617597071583096) q[3];
cx q[2],q[3];
rx(-0.1939254244053527) q[0];
rz(-0.027825065866338684) q[0];
rx(-0.02179561889843638) q[1];
rz(-0.2901199801923005) q[1];
rx(-0.18647569674027495) q[2];
rz(0.1893257414557701) q[2];
rx(-0.03247028595619648) q[3];
rz(-0.12975297847986572) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.15149560056719985) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.13154359922105857) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04054288835773504) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.36576174361605984) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.004222994440538553) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13953086980451726) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.10663564721479912) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.028850829094209097) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.039017384269610114) q[3];
cx q[2],q[3];
rx(-0.1995817861192831) q[0];
rz(-0.062021758286873646) q[0];
rx(-0.10566551159071998) q[1];
rz(-0.20152514556494985) q[1];
rx(-0.33325910130834246) q[2];
rz(0.14699286061503403) q[2];
rx(0.026023690908788755) q[3];
rz(-0.11857226590704917) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.22752646693315837) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11398269457734037) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08099158641623141) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.36850505433624614) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2201230926464518) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.25813797291976653) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.04435565248764765) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.03550321881690673) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07428470801417202) q[3];
cx q[2],q[3];
rx(-0.1778036181645325) q[0];
rz(-0.08559515360380136) q[0];
rx(-0.039562398581906744) q[1];
rz(-0.0514679536922966) q[1];
rx(-0.2946289976346064) q[2];
rz(0.07512778168780346) q[2];
rx(0.09821102034593553) q[3];
rz(-0.1359503245275727) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.21806338558615665) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.07526822786771936) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07569205830224461) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20876714076481104) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4253304219124119) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3864251028590735) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.05898125790300246) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0945386242920852) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.12323328566798131) q[3];
cx q[2],q[3];
rx(-0.13241942471105086) q[0];
rz(-0.13999501975232445) q[0];
rx(-0.06677984256377648) q[1];
rz(0.07744732319704167) q[1];
rx(-0.357526592140497) q[2];
rz(0.1658962604525069) q[2];
rx(0.16039709419552453) q[3];
rz(-0.13945005284591905) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08345832146446117) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11141072864290581) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03719358257809828) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.08525497342005653) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.48958424398268546) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4430681745047985) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.12853001878750223) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.02562653108498181) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.17523477493209946) q[3];
cx q[2],q[3];
rx(-0.08576342687067559) q[0];
rz(-0.11858248216456772) q[0];
rx(-0.0746591484919461) q[1];
rz(0.1298935611112213) q[1];
rx(-0.3388463063163195) q[2];
rz(0.2075932255921605) q[2];
rx(0.16583349710398068) q[3];
rz(-0.12743334644884743) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02587744992744835) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.014546638794874533) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07542685362910857) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12900156463434692) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5568632491302445) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5109508871382811) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.013383723259528416) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10899447257163382) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14194969833849735) q[3];
cx q[2],q[3];
rx(-0.07733203621009309) q[0];
rz(-0.17624215462859943) q[0];
rx(-0.08634040341797224) q[1];
rz(0.11010009068970626) q[1];
rx(-0.31267728833689207) q[2];
rz(0.18526602333074743) q[2];
rx(0.17218778321030473) q[3];
rz(-0.19425719155533175) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.032342665951612025) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0417808723980697) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.061641791469856884) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.02948021293418734) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.36930629549796085) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.34869098936423054) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12262719397563239) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.12610748809485395) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14368728425013366) q[3];
cx q[2],q[3];
rx(-0.03814744623026258) q[0];
rz(-0.22637460669758444) q[0];
rx(-0.05787139745158532) q[1];
rz(0.09151936190308574) q[1];
rx(-0.3328812500076835) q[2];
rz(0.18531614115897496) q[2];
rx(0.061775859807215346) q[3];
rz(-0.125366251306671) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02262447512502724) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09706383887411035) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.016116794219422862) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0208144651175616) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2799410703378501) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.23841349892012087) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1886409429344612) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.17611470463860213) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.14550646208782153) q[3];
cx q[2],q[3];
rx(-0.022585317528895418) q[0];
rz(-0.16814954258809883) q[0];
rx(-0.039893120042613625) q[1];
rz(-0.03741974529587453) q[1];
rx(-0.19863715033417934) q[2];
rz(0.13042376844330347) q[2];
rx(0.06197796198674145) q[3];
rz(-0.08312646203071103) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13500460668157285) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14744738706778968) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03690822652189307) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.031206203465690075) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19285701716881426) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05744021419687753) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.20379789720298433) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.15723023304645142) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1092564365808996) q[3];
cx q[2],q[3];
rx(-0.053253451798628164) q[0];
rz(-0.1634194601926825) q[0];
rx(0.028828761354966193) q[1];
rz(-0.030936164059068037) q[1];
rx(-0.07114689830016023) q[2];
rz(0.18913161077095927) q[2];
rx(0.005109304140195052) q[3];
rz(-0.05255452864381641) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.08237199947622414) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09892690629108124) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03367371856250954) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.006029253859363555) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.130887003565752) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05016083933414223) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1254878635915697) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11505579902408693) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05118886246700301) q[3];
cx q[2],q[3];
rx(-0.09022613973377196) q[0];
rz(-0.19410840217414693) q[0];
rx(0.048358825688537214) q[1];
rz(0.036454392019710194) q[1];
rx(-0.0023510407580585525) q[2];
rz(0.15573439426818558) q[2];
rx(-0.02401524627705468) q[3];
rz(-0.04086956058450875) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11978950193720925) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04299683316288916) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.015019947016595035) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.05528525304088851) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.047998363778437976) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.012420791464913263) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.08747557401805972) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07245514893229305) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.025751072243856452) q[3];
cx q[2],q[3];
rx(-0.04162860516684793) q[0];
rz(-0.2409808125269286) q[0];
rx(0.07852194167078994) q[1];
rz(0.04598087685728565) q[1];
rx(0.03961007500639222) q[2];
rz(0.13230310436324214) q[2];
rx(-0.06036047498319068) q[3];
rz(-0.0721775533535133) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.09297994757047855) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11216563412974644) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04647653137539814) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.08781473137684365) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.040180809094260424) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07755082301732694) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11161418397133749) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.08237904413441322) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.017871282000409456) q[3];
cx q[2],q[3];
rx(-0.12270774809045804) q[0];
rz(-0.1929742208193454) q[0];
rx(0.08263057738619167) q[1];
rz(0.024486973400354196) q[1];
rx(0.0882311125661508) q[2];
rz(0.08318790341205927) q[2];
rx(-0.04248622919299637) q[3];
rz(-0.09025625982275352) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.006815112081339312) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12689371051668444) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06346524140438763) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.18625716227494102) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.09574391352666693) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.058425514270235214) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07580297764444195) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.02264620442999482) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05434004128784752) q[3];
cx q[2],q[3];
rx(-0.18893882840485812) q[0];
rz(-0.20445564630869092) q[0];
rx(0.07992212610270145) q[1];
rz(-0.030426509115986794) q[1];
rx(0.010612872866812987) q[2];
rz(0.04645620284711183) q[2];
rx(-0.08651863382758697) q[3];
rz(-0.10901271501217341) q[3];