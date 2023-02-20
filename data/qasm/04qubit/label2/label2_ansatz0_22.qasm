OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.28808260158890253) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.20214331733799204) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01641250582380481) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7000437217342098) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.5277052602560812) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.010580781677327167) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4292492298755016) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.35166748332632725) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09816727577254782) q[3];
cx q[2],q[3];
rx(-0.029894100717202346) q[0];
rz(0.0650445117534588) q[0];
rx(0.045686657269967644) q[1];
rz(0.0010867113668662086) q[1];
rx(0.3704764498199622) q[2];
rz(-0.09575901570863911) q[2];
rx(-0.06660510497598539) q[3];
rz(-0.1371109917440401) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2693766865653613) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0662892035197432) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.050697945884553175) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.6246957415798567) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.44832286914058894) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0827494340951259) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2526164671946128) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.10538323210158039) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.018070525879791053) q[3];
cx q[2],q[3];
rx(-0.06896937777765373) q[0];
rz(0.0758099813858713) q[0];
rx(0.05306019507993155) q[1];
rz(0.11358186048724846) q[1];
rx(0.2158977673373944) q[2];
rz(0.03458215858552943) q[2];
rx(-0.017275486413566736) q[3];
rz(-0.2316507997675791) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2717810108594793) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.11107022577284723) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08806152919091398) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.42442109258830263) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.24152765052301864) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.045527929209754195) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05084279879072275) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.025530510769689995) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.033935326136764796) q[3];
cx q[2],q[3];
rx(-0.09268688716865361) q[0];
rz(0.09321190318257523) q[0];
rx(0.1147580011497735) q[1];
rz(0.07402560415265644) q[1];
rx(0.14440731598688583) q[2];
rz(0.04877164942196111) q[2];
rx(0.08211571184129292) q[3];
rz(-0.21056894634387924) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.25933356167493754) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03066780244492676) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028142787329080345) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.20468064446976078) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.09846548416766128) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.12665346841287947) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0737459332558302) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11299614644699314) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15141618909790497) q[3];
cx q[2],q[3];
rx(-0.07953290056342133) q[0];
rz(-0.021127643369233413) q[0];
rx(0.1328691235388523) q[1];
rz(0.04858908619272446) q[1];
rx(-0.21846724390624417) q[2];
rz(0.24721463603780633) q[2];
rx(0.17251618115825584) q[3];
rz(-0.16811933787264094) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.19461719227687474) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.05906002273567437) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.16811285659689604) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.08147331784568876) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.18015179356351627) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.07179838141652985) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.11578934717052596) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10236780002612218) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.032935686355686174) q[3];
cx q[2],q[3];
rx(-0.07402899626352857) q[0];
rz(-0.050248865438168165) q[0];
rx(0.14493040789725453) q[1];
rz(0.08726286296745077) q[1];
rx(-0.34738982559635034) q[2];
rz(0.2955400789472956) q[2];
rx(0.12105754888413485) q[3];
rz(-0.13668108989088348) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1050208655539503) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11373671732827749) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06731994351959238) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.14830665137465052) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.32019039811137495) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2844551612837419) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.012392716257059355) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.16156891776234567) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07592102518651145) q[3];
cx q[2],q[3];
rx(-0.1175129432933319) q[0];
rz(0.033027396520013) q[0];
rx(0.3127261141136728) q[1];
rz(0.10011006425362387) q[1];
rx(-0.5805987554064755) q[2];
rz(0.23523562734288875) q[2];
rx(0.1090460107014066) q[3];
rz(0.0032179984632901937) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.061914986495378146) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12866917072301035) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1849337320648645) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04716734585790915) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.43356134662774537) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4075612930890078) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.13706186001312426) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3248990485447993) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.18217454222621468) q[3];
cx q[2],q[3];
rx(-0.08952489761419284) q[0];
rz(-0.03020893582160375) q[0];
rx(0.3749435457090715) q[1];
rz(0.05081307033974089) q[1];
rx(-0.8364732183128153) q[2];
rz(0.07417308911894105) q[2];
rx(0.07395391153597505) q[3];
rz(0.15953189347296304) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07104873816282013) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1462863543296322) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3615837402421602) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.11457224776473644) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.5246814900578195) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.37762369385601685) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.12908571377053626) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.44182636098756506) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.00875098507876279) q[3];
cx q[2],q[3];
rx(-0.09222770889252908) q[0];
rz(-0.05153324503599052) q[0];
rx(0.31162717909202864) q[1];
rz(-0.09115119040327212) q[1];
rx(-0.8452203464233502) q[2];
rz(0.13453832054444256) q[2];
rx(-0.03524014437134166) q[3];
rz(0.18211099210182327) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07949197325662864) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1381591499057724) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.4737890253308832) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.18220322605807643) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.823065515279025) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.22537013472274667) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.14906655079938957) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.6405715215849932) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.006936034673056888) q[3];
cx q[2],q[3];
rx(-0.09938967825623533) q[0];
rz(-0.029575718592743924) q[0];
rx(0.42915197666912824) q[1];
rz(-0.2796682383934089) q[1];
rx(-0.841963852308415) q[2];
rz(0.16220711210385882) q[2];
rx(-0.13479871836259927) q[3];
rz(0.14884643069899392) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10436087456773352) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1286602990305444) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.4356179171628124) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.11362871933797447) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7390893444782286) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.26260648931407715) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.38976006319472084) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4750443137804076) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.12263627951459923) q[3];
cx q[2],q[3];
rx(-0.05121258850555926) q[0];
rz(-0.08365969370731045) q[0];
rx(0.42061325644186454) q[1];
rz(-0.3000907174967472) q[1];
rx(-0.9176372929557655) q[2];
rz(0.20166853077387237) q[2];
rx(-0.07193172906845195) q[3];
rz(0.03499111753310402) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0692400266961201) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13230493381775235) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3228622616438685) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1129648552199255) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6714946126555604) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5102678412342753) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.3694349134554775) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3714068261821895) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.28282521734294924) q[3];
cx q[2],q[3];
rx(-0.00816258631491103) q[0];
rz(-0.08106284796547517) q[0];
rx(0.3924602171367984) q[1];
rz(-0.11772790021065527) q[1];
rx(-0.7899940343734122) q[2];
rz(0.1332143966636143) q[2];
rx(-0.08570525187935292) q[3];
rz(-0.01711179951878658) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.055301402667042816) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.009520356325529956) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.16965368337080686) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.1421119647956746) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.8458976969981501) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5760124709322414) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.20973406764101826) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.45223704748342247) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2300005857430966) q[3];
cx q[2],q[3];
rx(-0.06128732755088958) q[0];
rz(-0.08933117523865193) q[0];
rx(0.4620682835019151) q[1];
rz(0.02919444616656125) q[1];
rx(-0.48342137806573227) q[2];
rz(0.06656943627130306) q[2];
rx(-0.007269844299611278) q[3];
rz(-0.0864242335323605) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.12142667788489436) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.007155159551396981) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07835836584137934) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.007261913876727586) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.8317333606792616) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5201230161922004) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.16263255469984111) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.44179242625975795) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.19709987680037014) q[3];
cx q[2],q[3];
rx(-0.05700051729362855) q[0];
rz(-0.17962855376999023) q[0];
rx(0.36426784541143153) q[1];
rz(0.023146548459097124) q[1];
rx(-0.41741497872759936) q[2];
rz(-0.024558524214722235) q[2];
rx(0.04957759707677217) q[3];
rz(-0.10088016327052013) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.14628572689752622) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.0009372653854012089) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04429009246581842) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04041534973975199) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.6212448030497983) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3218475546842714) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.02390831341403027) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.29036707079758606) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.11477495089358707) q[3];
cx q[2],q[3];
rx(-0.09859307423114602) q[0];
rz(-0.17315419293508572) q[0];
rx(0.13198131986238695) q[1];
rz(0.10419309773245891) q[1];
rx(-0.3489588305235254) q[2];
rz(-0.09271923005632673) q[2];
rx(-0.017891530318903604) q[3];
rz(-0.1213879702146269) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.2925085357765333) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.07435193479355329) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0027815554998644365) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.005275975200288146) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.39873020783855095) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.17286065000705936) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.11436713574327394) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11472274185154677) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0257009942298995) q[3];
cx q[2],q[3];
rx(-0.14492694408258724) q[0];
rz(-0.3768831928642836) q[0];
rx(-0.11478793680051548) q[1];
rz(0.04228137960254242) q[1];
rx(-0.25735694436900475) q[2];
rz(-0.11998326040124253) q[2];
rx(0.002902630862674766) q[3];
rz(-0.08724518100355473) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10343900664054925) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11574402516404278) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08262118601480416) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.028634021516681086) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.06002796698711227) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.059395604187983606) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2308270727000527) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.02102909899537733) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.061292748327556136) q[3];
cx q[2],q[3];
rx(-0.014198096383957436) q[0];
rz(-0.4149947755730399) q[0];
rx(-0.2883940355865127) q[1];
rz(-0.07747486363166185) q[1];
rx(-0.11126377765360974) q[2];
rz(-0.09276048942253991) q[2];
rx(0.07510600628751778) q[3];
rz(-0.044003254898535286) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.14150889710048942) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.054356773310297604) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.19296770969791885) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12591287366866427) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.058792704419685225) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.16261373698404247) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2937140702806414) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.051042522909159206) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.062187136431439964) q[3];
cx q[2],q[3];
rx(-0.005219407076822906) q[0];
rz(-0.44891179923142194) q[0];
rx(-0.4949514256390486) q[1];
rz(-0.1511260366935547) q[1];
rx(-0.05557997840740668) q[2];
rz(-0.10043304550140844) q[2];
rx(0.12841039968987922) q[3];
rz(0.040646824840557524) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.08501533519308138) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06550422996232992) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12037062779740121) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2178264079401555) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.017781759723538976) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15049243705680368) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2727965859999866) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.04225545902925953) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10654794916674996) q[3];
cx q[2],q[3];
rx(0.05702876176740732) q[0];
rz(-0.4417317747023237) q[0];
rx(-0.5909879101350124) q[1];
rz(-0.08955727411288171) q[1];
rx(-0.009653569680643419) q[2];
rz(-0.15478392307732153) q[2];
rx(0.07026048874694382) q[3];
rz(-0.004282234685720398) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.003685622516135888) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.08951002339119223) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.09248949689105596) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22880100918430346) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.09751543957494223) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.051314881424839344) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2723124980601991) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.08881924504685487) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1581468195221066) q[3];
cx q[2],q[3];
rx(0.14111219445409812) q[0];
rz(-0.2996742951586237) q[0];
rx(-0.6876657871408917) q[1];
rz(0.08414290638843343) q[1];
rx(0.08859289330712974) q[2];
rz(-0.250351165705019) q[2];
rx(0.010195854391042565) q[3];
rz(0.013704950358534647) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.016756349779898056) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13818456843945903) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.19675057852412764) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2009820170957178) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3398518666245481) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.17242296398675314) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14866950403851495) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.23813609361336885) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11436648443685478) q[3];
cx q[2],q[3];
rx(0.16843450014306557) q[0];
rz(-0.13873200129504587) q[0];
rx(-0.6186644021515263) q[1];
rz(0.014296553870710483) q[1];
rx(0.09060111312643072) q[2];
rz(-0.24605355323249561) q[2];
rx(0.009343034338635826) q[3];
rz(-0.013498656116439618) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.11068784347144119) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06708507959704657) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.16030498179454902) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.32210763022904093) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.5095426505728718) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2861268669619595) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07616305135623064) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.31871943256624286) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03272682024029275) q[3];
cx q[2],q[3];
rx(0.19493325211904322) q[0];
rz(-0.02168866528312689) q[0];
rx(-0.5843413616307448) q[1];
rz(0.016009067340995678) q[1];
rx(0.07556455609064225) q[2];
rz(-0.23300857537452016) q[2];
rx(0.12694147118111893) q[3];
rz(-0.04811743106381171) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.10729047242903135) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06933696363299774) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.11856154860183282) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4420260730779429) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3563403317476974) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3363441210285541) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.025083542937654177) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.21932299810869943) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03821330950595782) q[3];
cx q[2],q[3];
rx(0.18066662459318264) q[0];
rz(0.14365797496436988) q[0];
rx(-0.5572762318705426) q[1];
rz(-0.07614908617981594) q[1];
rx(0.0007060094699959218) q[2];
rz(-0.09494841448084548) q[2];
rx(0.11094050692422086) q[3];
rz(-0.03496826983691921) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09880800024627694) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2320833963418094) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1798334900270565) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3837748128662916) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1506452776136912) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3926920885888115) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.03554776835955053) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1857866690074212) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.015452076865525204) q[3];
cx q[2],q[3];
rx(0.15384319733787621) q[0];
rz(0.31684861681009113) q[0];
rx(-0.4506624422358098) q[1];
rz(-0.3273750498415289) q[1];
rx(0.022381155365534616) q[2];
rz(-0.05190147602801939) q[2];
rx(0.20723008969327614) q[3];
rz(-0.043961751667899944) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.026503171111239256) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.3610960172848372) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.3668123242401171) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3237557601914973) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11098293667163434) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2598445820650337) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.03328094802191776) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.034970386834943024) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07482680527072563) q[3];
cx q[2],q[3];
rx(0.09723748165451304) q[0];
rz(0.39716793861680727) q[0];
rx(-0.4073738975121834) q[1];
rz(-0.4703042548700509) q[1];
rx(-0.08473516140693158) q[2];
rz(-0.028883025559230203) q[2];
rx(0.18866657901387182) q[3];
rz(-0.025266679444405468) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03768249101224916) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.28227004019208185) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.262543032414818) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.3735019901182974) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.022262766107993562) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2680517201862085) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.019201501811937735) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.03394147069619928) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17044120078172806) q[3];
cx q[2],q[3];
rx(0.0025354415106179262) q[0];
rz(0.521860978975299) q[0];
rx(-0.31589970184217714) q[1];
rz(-0.6419162835570479) q[1];
rx(-0.13774628281565535) q[2];
rz(0.003046802330932321) q[2];
rx(0.16079215622423682) q[3];
rz(-0.09988270604101958) q[3];