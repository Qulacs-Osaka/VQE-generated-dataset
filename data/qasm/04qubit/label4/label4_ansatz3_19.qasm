OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.2796927802352505) q[0];
rz(-2.5912218423740465) q[0];
ry(-1.6727917471335023) q[1];
rz(1.7475249431886628) q[1];
ry(-2.9807851253075683) q[2];
rz(-1.7691102177154079) q[2];
ry(1.7300777617306045) q[3];
rz(2.1436439514726535) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3403913361818462) q[0];
rz(0.8949909712860976) q[0];
ry(2.1923417697622916) q[1];
rz(-1.0350135051158684) q[1];
ry(0.09978871990293024) q[2];
rz(0.954274213909863) q[2];
ry(1.5408672829186232) q[3];
rz(2.1208570170396075) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.8434812963320946) q[0];
rz(-1.030721605439571) q[0];
ry(0.8269683660361631) q[1];
rz(2.113938210327383) q[1];
ry(-2.365236278306176) q[2];
rz(2.746544444661805) q[2];
ry(-1.950565642373995) q[3];
rz(3.124247369349261) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.2242773321440095) q[0];
rz(-1.9857316707429886) q[0];
ry(-1.4430475941035894) q[1];
rz(0.7622155588185933) q[1];
ry(-2.748670794889809) q[2];
rz(-1.6458151610987137) q[2];
ry(-2.189123060772574) q[3];
rz(1.949838689457601) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7242793883982754) q[0];
rz(0.7254562389332233) q[0];
ry(-2.5828156216646145) q[1];
rz(1.6335865068015254) q[1];
ry(-0.15051758570994123) q[2];
rz(-1.5478144766546584) q[2];
ry(-2.3510251788043317) q[3];
rz(1.3807393511415424) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.4338444357298066) q[0];
rz(-0.7985110810973673) q[0];
ry(0.8430221038149526) q[1];
rz(2.018741561253334) q[1];
ry(-1.142593309584706) q[2];
rz(-0.5266627173682723) q[2];
ry(-1.5821900771296085) q[3];
rz(-3.1348695203240435) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.1747540827340623) q[0];
rz(2.3113017690757083) q[0];
ry(-2.5557685170801103) q[1];
rz(-0.4095859906963026) q[1];
ry(-1.5931899399030522) q[2];
rz(-1.5837588455736011) q[2];
ry(0.4570291908004594) q[3];
rz(0.4386663101960124) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.8735155735643894) q[0];
rz(-2.041153865686468) q[0];
ry(2.0759742651323227) q[1];
rz(2.754672151015772) q[1];
ry(-1.9940529478944962) q[2];
rz(0.33813409014752727) q[2];
ry(-1.8140259660834157) q[3];
rz(-0.07806676305430928) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2978423413739382) q[0];
rz(0.39844048366748286) q[0];
ry(2.307004535941421) q[1];
rz(2.4709692367040677) q[1];
ry(3.0508231817220586) q[2];
rz(-1.1842068503502805) q[2];
ry(2.974667598755235) q[3];
rz(2.9764739025852323) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.2755264690458743) q[0];
rz(-1.6349417732764109) q[0];
ry(-0.6657976162257082) q[1];
rz(-2.2077942054843023) q[1];
ry(-3.026918176902486) q[2];
rz(2.3164616848314914) q[2];
ry(0.4489692315976278) q[3];
rz(0.15604190403460305) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.4012767233465484) q[0];
rz(1.0803287686415748) q[0];
ry(-3.1056260501753363) q[1];
rz(-1.6154946126249659) q[1];
ry(-2.0504895546113953) q[2];
rz(1.2706630311318676) q[2];
ry(-0.8365942592532699) q[3];
rz(0.483642975054317) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8450370468458255) q[0];
rz(0.24868325378528658) q[0];
ry(1.5110838206913229) q[1];
rz(1.1371758543977903) q[1];
ry(1.5000876276814943) q[2];
rz(-0.5473735838938305) q[2];
ry(3.035916301597218) q[3];
rz(-1.4427369057190669) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.561258624427401) q[0];
rz(-0.907642704458878) q[0];
ry(-2.134119026619807) q[1];
rz(-0.0026631771239449264) q[1];
ry(1.1854490739243833) q[2];
rz(0.18566655793779877) q[2];
ry(1.2256397299622581) q[3];
rz(-1.8418711408034847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.3716991879849725) q[0];
rz(-1.029189887302462) q[0];
ry(-0.20009722324572596) q[1];
rz(-2.5600819400553463) q[1];
ry(2.829891164858028) q[2];
rz(-1.7853564530037156) q[2];
ry(1.622033513880341) q[3];
rz(1.905019720798518) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6152496003463186) q[0];
rz(-3.0464665645468796) q[0];
ry(-0.7415279724514727) q[1];
rz(0.5225612533281006) q[1];
ry(-1.2804318473062786) q[2];
rz(2.291281416405698) q[2];
ry(0.335137253878214) q[3];
rz(1.041128623599433) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.009289269000255706) q[0];
rz(1.5005267779885143) q[0];
ry(1.8403035239645096) q[1];
rz(2.4980690731978292) q[1];
ry(-0.08253255996942334) q[2];
rz(-0.1253486743851615) q[2];
ry(2.527597340387929) q[3];
rz(2.046633863489033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.4882681501665509) q[0];
rz(1.1175706994313477) q[0];
ry(-1.4314325092161433) q[1];
rz(-2.1308567381284993) q[1];
ry(0.06591710093195946) q[2];
rz(-1.9993633345234925) q[2];
ry(0.13253504007843417) q[3];
rz(1.4010643876448272) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.7063916831570745) q[0];
rz(-1.5154868436446236) q[0];
ry(3.1382140908891816) q[1];
rz(0.44068794859518956) q[1];
ry(-2.1092880386209956) q[2];
rz(1.5340229508168155) q[2];
ry(-1.9194944571328783) q[3];
rz(1.4762037411043438) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.973386952880282) q[0];
rz(-2.113940018551754) q[0];
ry(-0.34133055755270014) q[1];
rz(2.487183129925207) q[1];
ry(-3.1319496222208416) q[2];
rz(0.0868380083401425) q[2];
ry(-1.9886572067975115) q[3];
rz(2.4682505015762235) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.24845606839182074) q[0];
rz(0.1901013263186888) q[0];
ry(-2.8826235645824854) q[1];
rz(-0.39352059995208855) q[1];
ry(0.9297602091065305) q[2];
rz(2.9975083163494403) q[2];
ry(-3.0433557879159685) q[3];
rz(-1.400409084283351) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.2325852689568464) q[0];
rz(-3.120977968485136) q[0];
ry(2.9986539530172323) q[1];
rz(2.1287563211804974) q[1];
ry(-0.7082260280583353) q[2];
rz(-1.6729141713396132) q[2];
ry(2.003222347073943) q[3];
rz(2.0496918115411007) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.21703420047321167) q[0];
rz(-1.1815225611051945) q[0];
ry(-0.024053524525748493) q[1];
rz(-0.061543884687084424) q[1];
ry(1.674211819628745) q[2];
rz(-2.0659115388380824) q[2];
ry(3.036196363802929) q[3];
rz(-2.9439910680294847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7590963514716318) q[0];
rz(1.2827209792703054) q[0];
ry(-1.2611388669585306) q[1];
rz(-2.7633305003157704) q[1];
ry(-0.6597660876972881) q[2];
rz(0.486461325535144) q[2];
ry(-0.7629543934326949) q[3];
rz(1.2678999338424566) q[3];