OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.4041447600308149) q[0];
rz(-0.8943560925364675) q[0];
ry(-1.5564675187558217) q[1];
rz(0.3488249691215328) q[1];
ry(-1.8124140563768463) q[2];
rz(-1.469335397528643) q[2];
ry(-0.8708616175763265) q[3];
rz(-3.1092426919032956) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7931816341639513) q[0];
rz(1.9813161065637879) q[0];
ry(2.918871246021303) q[1];
rz(-2.1030677817663364) q[1];
ry(2.1322594232525875) q[2];
rz(0.5742029307478592) q[2];
ry(0.5297995436497575) q[3];
rz(1.453321193288983) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.796923297078055) q[0];
rz(-1.787277233876793) q[0];
ry(2.48844251903282) q[1];
rz(0.2976107100807104) q[1];
ry(-3.0880920510098195) q[2];
rz(1.9193992040625096) q[2];
ry(-2.0795543786502444) q[3];
rz(-3.0683507005536033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.086274583879569) q[0];
rz(-0.8310536246727643) q[0];
ry(1.7534937097279117) q[1];
rz(-1.2091641608927617) q[1];
ry(0.43746104771937944) q[2];
rz(-0.368964216637239) q[2];
ry(2.8752358454865425) q[3];
rz(0.574112918931923) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3090899674851917) q[0];
rz(-2.1669645603995384) q[0];
ry(1.7892256106106785) q[1];
rz(0.6353190825239441) q[1];
ry(1.9861593626093863) q[2];
rz(2.1055305010133667) q[2];
ry(-1.2578145739972424) q[3];
rz(-3.108277202137754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2532416736324459) q[0];
rz(-1.6037644268418294) q[0];
ry(-2.7980552841986253) q[1];
rz(-1.3531105257241656) q[1];
ry(1.4633437496978485) q[2];
rz(-2.0128538438063464) q[2];
ry(-1.874641196015582) q[3];
rz(0.9043421008769879) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.1900905465467226) q[0];
rz(1.8959708146951544) q[0];
ry(-0.07590082985003578) q[1];
rz(-0.10461559702275726) q[1];
ry(1.1667834330912354) q[2];
rz(0.46504944246775004) q[2];
ry(-1.4163467275194082) q[3];
rz(2.1418035051672457) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.5666603623410342) q[0];
rz(-0.9503101965867985) q[0];
ry(-0.6648284263969941) q[1];
rz(2.1870154393773937) q[1];
ry(0.08744904269950907) q[2];
rz(1.9410112964728752) q[2];
ry(2.6685449911373897) q[3];
rz(-2.4675239271632683) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8044713798790015) q[0];
rz(-0.3034676054451077) q[0];
ry(-0.34463788640007337) q[1];
rz(-2.5108024806520968) q[1];
ry(-1.544368833853341) q[2];
rz(-1.8081177232957797) q[2];
ry(-0.27254152076691973) q[3];
rz(1.4083927329212056) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.388114703365791) q[0];
rz(1.2605631529690617) q[0];
ry(2.2477242784144535) q[1];
rz(-0.28781551017564017) q[1];
ry(-0.10630608889624735) q[2];
rz(3.0335309744053762) q[2];
ry(-2.6770229991516685) q[3];
rz(1.6364575002698736) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.845846949620497) q[0];
rz(-0.5458690542765177) q[0];
ry(0.6380370171672471) q[1];
rz(3.0475384623960875) q[1];
ry(0.5369843548471733) q[2];
rz(2.420930785281337) q[2];
ry(2.562017943707068) q[3];
rz(-1.022232219752567) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2946957233309915) q[0];
rz(1.0403792477369729) q[0];
ry(0.8718250554138787) q[1];
rz(1.7705791595856724) q[1];
ry(-2.0177126919858575) q[2];
rz(-2.332144083962594) q[2];
ry(-1.9720276623334925) q[3];
rz(-2.8273926060745134) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.2524430015077077) q[0];
rz(1.6145786723630815) q[0];
ry(-1.5538199845338254) q[1];
rz(2.1441344330814402) q[1];
ry(-0.5275222747087046) q[2];
rz(3.055567158869339) q[2];
ry(-1.5790754528821693) q[3];
rz(0.8600215009394213) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8084727292521885) q[0];
rz(0.8412448785421718) q[0];
ry(0.045073484867225844) q[1];
rz(-1.1300540425344643) q[1];
ry(2.601116757770203) q[2];
rz(-2.6385668083376137) q[2];
ry(-2.0286585098621046) q[3];
rz(2.170974080296758) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.80795852300876) q[0];
rz(1.0729258634880823) q[0];
ry(-0.769240135588463) q[1];
rz(1.915181458469422) q[1];
ry(-0.8410662559268332) q[2];
rz(0.09343916278739785) q[2];
ry(-2.630852273432859) q[3];
rz(-0.3689422203128822) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.149475420538738) q[0];
rz(-0.7773905730084795) q[0];
ry(2.5122066604944098) q[1];
rz(0.8215691654980111) q[1];
ry(-1.2932757021823882) q[2];
rz(-0.4272812150083601) q[2];
ry(0.4446171651949956) q[3];
rz(-0.94645482475254) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8734001801544098) q[0];
rz(-0.32502420590557646) q[0];
ry(-0.03977047004207712) q[1];
rz(2.8381861095296763) q[1];
ry(-1.5084189079286903) q[2];
rz(0.7393114912232308) q[2];
ry(-1.8732358336453674) q[3];
rz(0.6135976817015479) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2835265152508937) q[0];
rz(0.9538408878409276) q[0];
ry(-2.9835251580898143) q[1];
rz(2.6345857937814854) q[1];
ry(0.8464305868304445) q[2];
rz(2.0174720537286848) q[2];
ry(-2.6995260610078935) q[3];
rz(-2.396786190957124) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.17340514877028365) q[0];
rz(0.9555703471104762) q[0];
ry(1.5350896879622598) q[1];
rz(1.5144284639593706) q[1];
ry(1.2012122857817948) q[2];
rz(2.091271949743331) q[2];
ry(-0.07172929306493087) q[3];
rz(-0.47855013810112146) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.4443083087983384) q[0];
rz(2.032108578551094) q[0];
ry(-2.2142168107155973) q[1];
rz(-0.5208513748338925) q[1];
ry(2.412963799287749) q[2];
rz(0.24999750900494536) q[2];
ry(-1.4800005139363093) q[3];
rz(2.2356047599891378) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4426640219817113) q[0];
rz(2.8646421645057973) q[0];
ry(-2.5018763010269645) q[1];
rz(-2.981808374076859) q[1];
ry(-2.268749681429426) q[2];
rz(-1.587094037379991) q[2];
ry(2.35164875790258) q[3];
rz(2.371916262351092) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.962273325646361) q[0];
rz(2.029602967505632) q[0];
ry(1.5742334859965956) q[1];
rz(0.039368211165897726) q[1];
ry(-1.1001077845202734) q[2];
rz(-0.8159072895104948) q[2];
ry(-0.3283523378612232) q[3];
rz(-1.111097504183937) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1851713148276017) q[0];
rz(-2.2907876928060915) q[0];
ry(-0.6572274587170828) q[1];
rz(-1.041851282952688) q[1];
ry(2.1813981644050466) q[2];
rz(-0.5416275368791785) q[2];
ry(3.050159222091351) q[3];
rz(-0.9591008846686968) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.660777016193138) q[0];
rz(0.7785421747841516) q[0];
ry(-0.971841777177144) q[1];
rz(-1.1756365840548373) q[1];
ry(0.6576604952035909) q[2];
rz(-0.7984140433307796) q[2];
ry(-2.94574475024697) q[3];
rz(-0.8422434474163973) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.287782982964572) q[0];
rz(-2.341580567905481) q[0];
ry(1.0246892338376803) q[1];
rz(2.9259749708535536) q[1];
ry(-1.8083564170761015) q[2];
rz(1.052911165268294) q[2];
ry(1.824642659233318) q[3];
rz(0.00825213829610494) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6647887948456305) q[0];
rz(2.5397926953008305) q[0];
ry(0.8544022317741795) q[1];
rz(-2.5826977368241675) q[1];
ry(-3.11281436730212) q[2];
rz(1.4247392493420437) q[2];
ry(-0.7246014592591444) q[3];
rz(1.6511289671795562) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7451273143777646) q[0];
rz(-2.1385132665306967) q[0];
ry(2.0897601786287687) q[1];
rz(2.570018094837029) q[1];
ry(0.217987313446538) q[2];
rz(-1.8356488064198133) q[2];
ry(0.23586516983270928) q[3];
rz(-1.1582491790436127) q[3];