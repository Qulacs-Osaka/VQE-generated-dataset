OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.5986622861937193) q[0];
rz(-0.8052300224014519) q[0];
ry(-1.1630813358369299) q[1];
rz(0.5590881638199239) q[1];
ry(-2.5138531018157657) q[2];
rz(-2.3702955547285556) q[2];
ry(3.134564100027156) q[3];
rz(3.050258315467387) q[3];
ry(1.0955827837397905) q[4];
rz(-0.8684741614140303) q[4];
ry(-0.11224817970167907) q[5];
rz(2.463100434055453) q[5];
ry(-0.020584507728854413) q[6];
rz(-0.7084193723959834) q[6];
ry(-0.15642574622661226) q[7];
rz(-0.0072093416237350985) q[7];
ry(3.1155628611796837) q[8];
rz(-0.3733362243774634) q[8];
ry(-1.505724726536748) q[9];
rz(0.35802907951814605) q[9];
ry(1.547415000825109) q[10];
rz(0.6339917343790956) q[10];
ry(3.136349860663261) q[11];
rz(-0.4359383336292275) q[11];
ry(1.850080706941573) q[12];
rz(1.4236002999228785) q[12];
ry(-0.03288900445667622) q[13];
rz(0.0957728196518836) q[13];
ry(-3.140950870384574) q[14];
rz(2.517111923701229) q[14];
ry(3.1034281251424227) q[15];
rz(0.46838587966743234) q[15];
ry(-2.3967656994798925) q[16];
rz(1.0004048979644233) q[16];
ry(1.4979279367469367) q[17];
rz(-2.017423746391347) q[17];
ry(-0.892333137994774) q[18];
rz(-1.6859581569676791) q[18];
ry(1.9205674696919237) q[19];
rz(1.042278761359647) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.6721806716515353) q[0];
rz(1.4986955439042964) q[0];
ry(2.1495348982088016) q[1];
rz(-1.2160245891201171) q[1];
ry(-1.3599216629823472) q[2];
rz(2.07864133733963) q[2];
ry(0.060922939574338086) q[3];
rz(2.9625647385623926) q[3];
ry(-2.061141883941221) q[4];
rz(1.8081039689191103) q[4];
ry(0.019705774880019872) q[5];
rz(2.2780719601359083) q[5];
ry(0.07639418171150147) q[6];
rz(-3.131022927241916) q[6];
ry(3.089442757016744) q[7];
rz(2.569490078839006) q[7];
ry(-2.982440673107587) q[8];
rz(-1.6340961034484767) q[8];
ry(-1.7812148937823844) q[9];
rz(-2.086442066126354) q[9];
ry(0.6845934833120044) q[10];
rz(-1.1885573810533754) q[10];
ry(1.8403408018439265) q[11];
rz(-1.968638185423738) q[11];
ry(-1.2467633646376242) q[12];
rz(1.4219946929188416) q[12];
ry(0.6447817804960536) q[13];
rz(0.8003138291972239) q[13];
ry(3.133534597028064) q[14];
rz(3.006951290114527) q[14];
ry(-2.967080706900877) q[15];
rz(-2.803810871273984) q[15];
ry(2.733360253184228) q[16];
rz(-1.8412108835695016) q[16];
ry(-0.7346577360889635) q[17];
rz(1.4727971418198003) q[17];
ry(2.089447459647709) q[18];
rz(0.0233691139953911) q[18];
ry(2.9088743210306056) q[19];
rz(-1.096419455894905) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.4117550551918283) q[0];
rz(-0.2134596193426913) q[0];
ry(1.8747369337898547) q[1];
rz(-0.2713428705425394) q[1];
ry(2.661066300518502) q[2];
rz(0.18763124084526783) q[2];
ry(1.6139747517316545) q[3];
rz(-1.651151359018348) q[3];
ry(2.387641761201825) q[4];
rz(-0.23458973870130526) q[4];
ry(-1.938130260628327) q[5];
rz(-1.7743602319381164) q[5];
ry(1.9128668813256238) q[6];
rz(0.9693167105386458) q[6];
ry(-1.4831317189150464) q[7];
rz(-2.8223879166484505) q[7];
ry(0.049357615657001475) q[8];
rz(1.275278897306918) q[8];
ry(0.015207342776326627) q[9];
rz(0.5565027885302776) q[9];
ry(-0.3849750089111392) q[10];
rz(-2.5995486711721973) q[10];
ry(0.005487890969110902) q[11];
rz(0.7745766555281363) q[11];
ry(-0.026660298247184855) q[12];
rz(-0.13233609291173032) q[12];
ry(-3.0481325275530913) q[13];
rz(-2.5005010354001254) q[13];
ry(-0.023024247723853364) q[14];
rz(-2.2022260125267237) q[14];
ry(0.17143546530372317) q[15];
rz(2.2648451021427256) q[15];
ry(-2.4070098300290845) q[16];
rz(2.9240508360757387) q[16];
ry(2.259724596820111) q[17];
rz(-1.2254305873238018) q[17];
ry(-0.5091851888469362) q[18];
rz(-0.5879083019382092) q[18];
ry(-2.356800534731498) q[19];
rz(0.3062762537833896) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.07727598006037173) q[0];
rz(1.4283905637869907) q[0];
ry(-1.9551997042606102) q[1];
rz(-2.086913198659235) q[1];
ry(3.127458554994508) q[2];
rz(-1.7654387514890868) q[2];
ry(-0.4559846033512063) q[3];
rz(1.8685077155644727) q[3];
ry(0.00026853202224856306) q[4];
rz(2.649816228748197) q[4];
ry(1.3991943574068224) q[5];
rz(-1.4094085762016824) q[5];
ry(-3.1218552320231696) q[6];
rz(-0.8169124869081162) q[6];
ry(3.122130932173699) q[7];
rz(2.059519885247848) q[7];
ry(-0.004202217737885914) q[8];
rz(1.8501692495440771) q[8];
ry(-3.1009008317190196) q[9];
rz(0.2069663411660827) q[9];
ry(0.7480902203494593) q[10];
rz(-0.4690492601804923) q[10];
ry(2.61583217432319) q[11];
rz(1.9407695457497502) q[11];
ry(1.671886134318695) q[12];
rz(1.8892025319017316) q[12];
ry(-2.951467582639855) q[13];
rz(1.3141572881235417) q[13];
ry(-3.0946469727822747) q[14];
rz(0.6861986017187726) q[14];
ry(-0.007298332478234229) q[15];
rz(-0.9806299294289701) q[15];
ry(-0.9079113612726468) q[16];
rz(2.016120620847306) q[16];
ry(-1.8504351814473772) q[17];
rz(2.35872932782148) q[17];
ry(2.5111429331301642) q[18];
rz(-1.3531752583198267) q[18];
ry(2.546113453765348) q[19];
rz(2.923931474990612) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.05340403596546442) q[0];
rz(1.8919247340980947) q[0];
ry(-0.8225979619776558) q[1];
rz(1.4053047464167574) q[1];
ry(1.5934498856624764) q[2];
rz(2.376000294243079) q[2];
ry(-2.9915319597019536) q[3];
rz(2.500823799999448) q[3];
ry(-3.0936189583131544) q[4];
rz(2.4009565654850675) q[4];
ry(-1.5561899944619946) q[5];
rz(-1.9378530946115482) q[5];
ry(-0.47914676495003317) q[6];
rz(-1.5531482539042163) q[6];
ry(-2.796309379379379) q[7];
rz(-0.007168183429146722) q[7];
ry(1.6065623890377836) q[8];
rz(1.5104946105135788) q[8];
ry(1.654328317685252) q[9];
rz(0.03881785375780071) q[9];
ry(1.234586909126372) q[10];
rz(-1.561256655712051) q[10];
ry(1.4394223491087361) q[11];
rz(-3.011487031230792) q[11];
ry(1.5605075863786775) q[12];
rz(1.591684644378792) q[12];
ry(0.3291688641558465) q[13];
rz(0.6766498573553491) q[13];
ry(-1.3784385274012019) q[14];
rz(0.6847334509036918) q[14];
ry(0.11925978335140819) q[15];
rz(-1.9112063363504637) q[15];
ry(1.863930990778412) q[16];
rz(0.5108810140079036) q[16];
ry(-0.8313170060633251) q[17];
rz(1.016352675765367) q[17];
ry(1.2452931268007918) q[18];
rz(2.074810039827358) q[18];
ry(0.8032815244781076) q[19];
rz(-0.26469068778650445) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5273433538275913) q[0];
rz(-1.5171425154259195) q[0];
ry(-1.6862938210321614) q[1];
rz(2.6513922579234848) q[1];
ry(-0.01650317726284909) q[2];
rz(3.0486904293218364) q[2];
ry(0.010645428394146172) q[3];
rz(0.921844774366118) q[3];
ry(-2.9322051071741035) q[4];
rz(1.5539042701514452) q[4];
ry(2.8081508103684367) q[5];
rz(0.2236967893202625) q[5];
ry(-1.569486560276097) q[6];
rz(-1.5586690578810833) q[6];
ry(0.00046363973345036344) q[7];
rz(1.5747631217447953) q[7];
ry(3.0861574022551137) q[8];
rz(-1.6380582730154334) q[8];
ry(3.1186607490125002) q[9];
rz(1.5703256029120887) q[9];
ry(-0.11646972643786997) q[10];
rz(0.0022646526463203647) q[10];
ry(2.3978111594877745) q[11];
rz(1.9519306085471664) q[11];
ry(-0.9276850522975737) q[12];
rz(1.5520574109461511) q[12];
ry(-1.3171826957345705) q[13];
rz(-3.136258563787003) q[13];
ry(-0.021185798274784725) q[14];
rz(3.0697127746141377) q[14];
ry(3.138095787354556) q[15];
rz(2.814905326354775) q[15];
ry(-1.002250566375474) q[16];
rz(2.5952445542181586) q[16];
ry(1.7418980808027822) q[17];
rz(-1.9414403828559736) q[17];
ry(-0.6926239797291114) q[18];
rz(1.1178453242244586) q[18];
ry(1.4051691507526947) q[19];
rz(-0.23821165584679108) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.3415830510648172) q[0];
rz(-1.2504363629376138) q[0];
ry(-0.25847751907554806) q[1];
rz(2.170646002841493) q[1];
ry(2.562626160773739) q[2];
rz(-2.8699327466639857) q[2];
ry(-1.6961296307022533) q[3];
rz(-0.41197781641305004) q[3];
ry(-1.572718166811733) q[4];
rz(-1.4829907294555198) q[4];
ry(-1.571246393776919) q[5];
rz(1.5827551532476534) q[5];
ry(-1.5703897400818958) q[6];
rz(-2.8189774603438504) q[6];
ry(-0.29310787972287417) q[7];
rz(-2.9968885972707024) q[7];
ry(-1.5669592686176166) q[8];
rz(2.987559946871426) q[8];
ry(1.7068405388773247) q[9];
rz(2.6419329773096187) q[9];
ry(1.797252920805088) q[10];
rz(-0.3162539755497984) q[10];
ry(0.016256605534226587) q[11];
rz(-0.10468823001938271) q[11];
ry(-3.1068714528807706) q[12];
rz(-1.7012823547407339) q[12];
ry(1.5535971858342403) q[13];
rz(0.00417068674098926) q[13];
ry(-0.007027860811697816) q[14];
rz(2.556536218008808) q[14];
ry(-3.117258542157606) q[15];
rz(-2.3649324571228068) q[15];
ry(-2.2311431874871) q[16];
rz(0.4186277267943054) q[16];
ry(-0.2733487838797672) q[17];
rz(-2.872274949740532) q[17];
ry(0.8298495646852737) q[18];
rz(1.5079965235807922) q[18];
ry(1.775980875614013) q[19];
rz(2.6476957772243885) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.8248085043921096) q[0];
rz(-1.508443933562468) q[0];
ry(-2.588149056313772) q[1];
rz(-1.378372490863965) q[1];
ry(-3.123400663348453) q[2];
rz(-1.4570297746192242) q[2];
ry(-1.5798142773821242) q[3];
rz(-1.4474900481594197) q[3];
ry(1.708118784988596) q[4];
rz(-3.0006872040857724) q[4];
ry(1.564787572120484) q[5];
rz(0.2731766351215532) q[5];
ry(0.06420267143416268) q[6];
rz(1.9577064421602914) q[6];
ry(-3.048701727387881) q[7];
rz(0.12130901275858529) q[7];
ry(-2.7877071355714187) q[8];
rz(2.949942667540545) q[8];
ry(-1.772767050712348) q[9];
rz(-0.04762834452348532) q[9];
ry(0.3270519971989261) q[10];
rz(-2.7909905649247158) q[10];
ry(-0.46444452131449865) q[11];
rz(-1.5559692477992564) q[11];
ry(3.0759306625849954) q[12];
rz(2.012576937688636) q[12];
ry(1.3144775221005969) q[13];
rz(-2.5357262623761008) q[13];
ry(1.5830917604164212) q[14];
rz(1.8219152520144926) q[14];
ry(1.5414201193000505) q[15];
rz(-0.11083884194028748) q[15];
ry(-0.6942578636964836) q[16];
rz(-1.0182404034842023) q[16];
ry(1.6519462965300051) q[17];
rz(0.22515480871694837) q[17];
ry(1.586688249018457) q[18];
rz(1.6513420377100863) q[18];
ry(1.2650129857676509) q[19];
rz(-3.0387691200711955) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.8542409091700085) q[0];
rz(-2.196756724684686) q[0];
ry(-2.4881964580963274) q[1];
rz(-2.313189499123965) q[1];
ry(0.0033347764665474246) q[2];
rz(-0.6373358533789119) q[2];
ry(1.5548934931765404) q[3];
rz(-1.5755781943601441) q[3];
ry(1.6154761612599866) q[4];
rz(-1.7500404714392244) q[4];
ry(-1.5291631872976654) q[5];
rz(-1.7182047724860183) q[5];
ry(-3.1242974949116156) q[6];
rz(1.299177992561871) q[6];
ry(0.3314616901915146) q[7];
rz(2.868010024377942) q[7];
ry(0.00904231004401424) q[8];
rz(0.0857247451608565) q[8];
ry(-0.03094019259856861) q[9];
rz(0.7288416571582247) q[9];
ry(-0.31623085974802473) q[10];
rz(-0.043039964045626156) q[10];
ry(-0.003814277263484378) q[11];
rz(1.2386649027085388) q[11];
ry(3.130613297090097) q[12];
rz(-3.118731574915795) q[12];
ry(-0.02983664739983816) q[13];
rz(2.639635577473486) q[13];
ry(1.438398883321356) q[14];
rz(-2.4743196459057484) q[14];
ry(0.05281261893229705) q[15];
rz(-3.0462349919552314) q[15];
ry(3.140142237874188) q[16];
rz(-1.3475543472215863) q[16];
ry(0.98104894041025) q[17];
rz(2.2062169653810813) q[17];
ry(-1.5481741379357368) q[18];
rz(-2.6088569577072445) q[18];
ry(-3.08202375789979) q[19];
rz(0.3515082989462313) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.5351875398510457) q[0];
rz(1.60606868106666) q[0];
ry(1.825138030673263) q[1];
rz(-1.1088484050565715) q[1];
ry(0.007874262340000548) q[2];
rz(0.13323909683302115) q[2];
ry(-1.5873896409528276) q[3];
rz(2.120403707548963) q[3];
ry(0.019112736549177) q[4];
rz(0.053820834435309856) q[4];
ry(-3.139036202569217) q[5];
rz(0.12591529969613188) q[5];
ry(-0.45903429156857134) q[6];
rz(1.4493521060987566) q[6];
ry(1.2061416530955622) q[7];
rz(0.08694313423280065) q[7];
ry(1.7004811464567222) q[8];
rz(-1.2143898347063908) q[8];
ry(0.1874438942439033) q[9];
rz(-2.388633038344874) q[9];
ry(-0.8031281404482137) q[10];
rz(-2.2865093665768406) q[10];
ry(0.4811146447536636) q[11];
rz(0.022185547985124288) q[11];
ry(-3.1407765814984487) q[12];
rz(-1.595970373551908) q[12];
ry(2.9894568910586776) q[13];
rz(-0.007050187769414825) q[13];
ry(3.122512614947235) q[14];
rz(-0.8817736080980413) q[14];
ry(1.5386783265061605) q[15];
rz(-0.7200060086220725) q[15];
ry(-1.0833077779032534) q[16];
rz(-1.7082864830600146) q[16];
ry(-2.7199390443979197) q[17];
rz(1.3244750293276262) q[17];
ry(1.1528280859690625) q[18];
rz(1.9556915609838974) q[18];
ry(3.134249227459693) q[19];
rz(0.9575298742357967) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.7465933204739175) q[0];
rz(0.6494535208638861) q[0];
ry(0.9144612885726477) q[1];
rz(-1.2548668416000786) q[1];
ry(0.0021633050359941564) q[2];
rz(1.6892012228653517) q[2];
ry(-0.23956728772795882) q[3];
rz(2.5712429355319006) q[3];
ry(1.56802763376837) q[4];
rz(1.5761944019589293) q[4];
ry(-1.7217821783920604) q[5];
rz(2.209150663429642) q[5];
ry(3.0738753391337252) q[6];
rz(3.101602995574233) q[6];
ry(3.0449731761497394) q[7];
rz(-3.004063298470738) q[7];
ry(3.1408943136762653) q[8];
rz(1.883102070580317) q[8];
ry(2.869516899989117) q[9];
rz(-0.022930697299518243) q[9];
ry(3.0993373592488713) q[10];
rz(0.7018239826261129) q[10];
ry(1.5625923377874338) q[11];
rz(-3.1170017483813828) q[11];
ry(0.03279966850773697) q[12];
rz(-0.5126643021567663) q[12];
ry(-0.058007315709345794) q[13];
rz(-1.822011992117842) q[13];
ry(-1.407350024670797) q[14];
rz(-0.9678258497649271) q[14];
ry(3.1118147332777153) q[15];
rz(-2.290048420358014) q[15];
ry(-1.3616668123447884) q[16];
rz(3.1381869658550063) q[16];
ry(1.8736388791321108) q[17];
rz(-3.0976213660780445) q[17];
ry(0.3854046409570513) q[18];
rz(1.1252577139272297) q[18];
ry(-1.0926305946446648) q[19];
rz(-0.7463042616916278) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5275170790733092) q[0];
rz(-1.0704482196432097) q[0];
ry(2.1784861473767005) q[1];
rz(-1.728719371494738) q[1];
ry(2.1743370231962977) q[2];
rz(-1.97742928674547) q[2];
ry(1.4451532294563671) q[3];
rz(0.02229216817026842) q[3];
ry(-1.5680909313177682) q[4];
rz(-0.014018830040947986) q[4];
ry(-0.005404363671435049) q[5];
rz(1.0318300600990435) q[5];
ry(1.5669249934030685) q[6];
rz(1.5996152581595646) q[6];
ry(1.4595970337580182) q[7];
rz(-0.05676475744190944) q[7];
ry(3.087456765411843) q[8];
rz(-2.7383620443699677) q[8];
ry(2.729643934651824) q[9];
rz(-1.8358584338511124) q[9];
ry(-0.0017587141144859086) q[10];
rz(0.2058821303504725) q[10];
ry(1.5078249823747578) q[11];
rz(-0.5416903087011731) q[11];
ry(-1.5844144254095958) q[12];
rz(3.131386906492851) q[12];
ry(-1.5118148586712845) q[13];
rz(-3.1404551000111294) q[13];
ry(0.0003731137517988259) q[14];
rz(-2.4346623521457573) q[14];
ry(-1.5880319699015437) q[15];
rz(1.7061565130812557) q[15];
ry(0.23896231245622682) q[16];
rz(0.0030476271721973096) q[16];
ry(-7.948834106485946e-07) q[17];
rz(-0.8837891358439313) q[17];
ry(1.6053782031076858) q[18];
rz(-1.9996706985045012) q[18];
ry(1.0479022952356134) q[19];
rz(-1.669232230550832) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.016551059228184) q[0];
rz(1.8310767418390288) q[0];
ry(0.013173989363892693) q[1];
rz(1.1812378788178017) q[1];
ry(-3.0675744355164944) q[2];
rz(-1.9687759098772775) q[2];
ry(-0.1679668263230365) q[3];
rz(-0.5076507683861011) q[3];
ry(1.9770176933747372) q[4];
rz(0.04668465110860164) q[4];
ry(1.682884639969389) q[5];
rz(1.6162900116509684) q[5];
ry(1.524421649512334) q[6];
rz(-0.4600116262800392) q[6];
ry(0.07485390853922037) q[7];
rz(-0.04774093007852061) q[7];
ry(1.5779742357753612) q[8];
rz(-3.132672430587285) q[8];
ry(3.052273890575667) q[9];
rz(1.0871757106685704) q[9];
ry(-3.0968392589918317) q[10];
rz(1.5603471322051814) q[10];
ry(-0.006998444400038429) q[11];
rz(0.5637933230841679) q[11];
ry(-1.5742645208087227) q[12];
rz(1.1343790103611644) q[12];
ry(-0.29906235414750704) q[13];
rz(-3.0239907781317386) q[13];
ry(0.0013689390409883373) q[14];
rz(-1.3610318276583189) q[14];
ry(-2.794787195112079) q[15];
rz(-3.090263829603247) q[15];
ry(-1.555195967900312) q[16];
rz(1.2274522868721334) q[16];
ry(0.9373878456695007) q[17];
rz(-0.918998208134175) q[17];
ry(1.5607892544830166) q[18];
rz(-0.0544042025073889) q[18];
ry(1.4174874617402917) q[19];
rz(3.0954171104501302) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.8684981029305199) q[0];
rz(2.980106894500925) q[0];
ry(-0.030316524895260738) q[1];
rz(2.136735644105218) q[1];
ry(-2.202266598966978) q[2];
rz(0.33144170541653595) q[2];
ry(-0.10176997949796948) q[3];
rz(-1.8437171176625275) q[3];
ry(0.007565657676612083) q[4];
rz(-0.04813420314413097) q[4];
ry(-2.641543732315046) q[5];
rz(-2.5399783205934465) q[5];
ry(-3.1189718943327813) q[6];
rz(2.6688653061689114) q[6];
ry(1.4403592400265648) q[7];
rz(-3.13922489237717) q[7];
ry(-1.4231631626045305) q[8];
rz(-1.1804584418296704) q[8];
ry(-1.573669888422674) q[9];
rz(0.0062336038712780244) q[9];
ry(-2.8190798996578703) q[10];
rz(-3.1244930171795984) q[10];
ry(-2.7582288201664795) q[11];
rz(1.9047819993841273) q[11];
ry(-0.03799695625957078) q[12];
rz(-2.630054503443244) q[12];
ry(0.03881164565695534) q[13];
rz(3.111546097438309) q[13];
ry(-0.0014005383190338172) q[14];
rz(-1.5220916184118813) q[14];
ry(1.3111812367106592) q[15];
rz(-2.1794786668456023) q[15];
ry(-3.1136477810211973) q[16];
rz(-1.32415541111118) q[16];
ry(-1.5733360733661774) q[17];
rz(0.665480546649011) q[17];
ry(1.9697381984159226) q[18];
rz(-2.719276505057329) q[18];
ry(-1.3411414982534424) q[19];
rz(2.472364318865149) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.6089517438233587) q[0];
rz(-1.3286390835860464) q[0];
ry(1.6281694174036019) q[1];
rz(1.3985050028359831) q[1];
ry(-0.08621893396035911) q[2];
rz(-1.9233880330597521) q[2];
ry(0.027601783358180795) q[3];
rz(-0.8516182280289007) q[3];
ry(-1.537108466527167) q[4];
rz(-1.2308081544003224) q[4];
ry(3.1274993975921124) q[5];
rz(-2.541576424970039) q[5];
ry(1.574753685603017) q[6];
rz(-0.013111253528066413) q[6];
ry(1.7327822850484396) q[7];
rz(-3.1059125027490335) q[7];
ry(-3.141355783063705) q[8];
rz(-1.1760259125100094) q[8];
ry(1.368111999944741) q[9];
rz(3.1232249462284005) q[9];
ry(-1.567017805274324) q[10];
rz(-3.13945600998924) q[10];
ry(-0.00491257659636465) q[11];
rz(0.2918337552274632) q[11];
ry(-0.0007327834941233019) q[12];
rz(1.5029826856793456) q[12];
ry(-0.08771131294513511) q[13];
rz(3.0711808460686396) q[13];
ry(0.003433423987168283) q[14];
rz(-1.5333088337243792) q[14];
ry(-2.7070441103041585) q[15];
rz(-2.440003025082569) q[15];
ry(2.861227694908828) q[16];
rz(-0.24732354454082997) q[16];
ry(-0.01055150527733086) q[17];
rz(-2.196920710435257) q[17];
ry(1.6070061476392272) q[18];
rz(0.05884728159194186) q[18];
ry(-0.9087846777534923) q[19];
rz(1.6935006784118443) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.5183627105871649) q[0];
rz(0.02702324814632867) q[0];
ry(-2.7639464968414424) q[1];
rz(-3.0474907007412284) q[1];
ry(0.11948471411784212) q[2];
rz(-1.0247176462065746) q[2];
ry(-0.6348164613794136) q[3];
rz(3.1353590216495824) q[3];
ry(-3.1317400930817048) q[4];
rz(-1.2266241960400062) q[4];
ry(1.5656480234528771) q[5];
rz(2.825203274194537) q[5];
ry(-0.5668684769034532) q[6];
rz(-2.2914641197322148) q[6];
ry(3.0855595711564487) q[7];
rz(0.042539383351141595) q[7];
ry(0.7558846393654446) q[8];
rz(0.0031758892900054145) q[8];
ry(-3.1358023415866083) q[9];
rz(3.112200822462156) q[9];
ry(-1.3967837427400895) q[10];
rz(0.9700954726765625) q[10];
ry(-0.004892428236564172) q[11];
rz(-1.3557460407536768) q[11];
ry(1.5268976037871203) q[12];
rz(3.105670200688506) q[12];
ry(1.5096941519714981) q[13];
rz(-0.1185541209161546) q[13];
ry(1.5731416762526012) q[14];
rz(1.1134504784706993) q[14];
ry(3.1382913294114094) q[15];
rz(2.096735084552851) q[15];
ry(0.058371589588284464) q[16];
rz(1.7617924107356624) q[16];
ry(-0.0075622479568227974) q[17];
rz(-1.6219965982750466) q[17];
ry(-2.249489621988709) q[18];
rz(0.03331934845191675) q[18];
ry(-1.63428117821704) q[19];
rz(1.8476322068264972) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.1306789846635708) q[0];
rz(1.7859186090431605) q[0];
ry(2.678023940233594) q[1];
rz(1.869190838887387) q[1];
ry(-1.5951116182538572) q[2];
rz(-3.137749588790209) q[2];
ry(0.024299204102625843) q[3];
rz(0.006764789766049439) q[3];
ry(1.5724113921164107) q[4];
rz(-2.0698165720841577) q[4];
ry(-3.1320697399246757) q[5];
rz(2.9372874551614196) q[5];
ry(0.0014435208086949203) q[6];
rz(2.2894937714195316) q[6];
ry(-0.5036438372100882) q[7];
rz(3.13298540688463) q[7];
ry(2.7642196127130445) q[8];
rz(0.002740338589599082) q[8];
ry(-2.8020852634751643) q[9];
rz(-0.0030874213997158506) q[9];
ry(3.1393656047599054) q[10];
rz(-2.1493271198129085) q[10];
ry(0.0007211655249639293) q[11];
rz(-0.8407744766360177) q[11];
ry(1.6438357667860106) q[12];
rz(-0.040507072906879486) q[12];
ry(-3.059625953010517) q[13];
rz(2.932956593914501) q[13];
ry(0.6225427547903095) q[14];
rz(-2.8184944482328085) q[14];
ry(-0.001847925416826729) q[15];
rz(1.1907923299211038) q[15];
ry(0.18374841204983275) q[16];
rz(-0.9598467671900243) q[16];
ry(2.8538326824184015) q[17];
rz(-0.001210162612453658) q[17];
ry(-0.10414671860247715) q[18];
rz(1.395216698584478) q[18];
ry(2.872167169511435) q[19];
rz(-2.524644267046037) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.4008063655626968) q[0];
rz(1.4763121145760234) q[0];
ry(-1.5729145037793895) q[1];
rz(1.5046931971501083) q[1];
ry(-1.6574198275975078) q[2];
rz(0.014427927809927077) q[2];
ry(-1.5683799148811097) q[3];
rz(3.1377392293385595) q[3];
ry(-0.0003667378644429675) q[4];
rz(0.7957408116846506) q[4];
ry(3.0927304353384986) q[5];
rz(0.12628331151420724) q[5];
ry(-0.6526995959062099) q[6];
rz(-0.00016384407265856996) q[6];
ry(1.58147534039007) q[7];
rz(-3.136243773509888) q[7];
ry(-2.3276233958246344) q[8];
rz(1.4174950284892385) q[8];
ry(1.5736295464312766) q[9];
rz(-1.2614970835866064) q[9];
ry(1.2993984053255714) q[10];
rz(2.0212944763736864) q[10];
ry(-2.7878559891655676) q[11];
rz(2.854191964936094) q[11];
ry(-3.119978091231614) q[12];
rz(3.105300495652088) q[12];
ry(3.1281684039558066) q[13];
rz(-0.07968214456434099) q[13];
ry(3.136582078663111) q[14];
rz(-0.3481692526584345) q[14];
ry(-3.140113359164028) q[15];
rz(-1.4515373007631087) q[15];
ry(1.6143699324758227) q[16];
rz(-1.0747605188635496) q[16];
ry(-1.5389357912049366) q[17];
rz(-0.2515035291669703) q[17];
ry(0.04959859876157058) q[18];
rz(1.405645422498927) q[18];
ry(2.8293020745846436) q[19];
rz(-0.30242988075301996) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.575443513510593) q[0];
rz(-0.008445156652990345) q[0];
ry(0.16412052885466813) q[1];
rz(1.6225055530962473) q[1];
ry(-1.5758260382994362) q[2];
rz(-3.107918257795176) q[2];
ry(-1.7988386393511897) q[3];
rz(-3.070482459944564) q[3];
ry(1.0071247465622513) q[4];
rz(0.0603387040721719) q[4];
ry(-2.606244132687952) q[5];
rz(3.1295002133569856) q[5];
ry(0.32715238584099166) q[6];
rz(-3.1355450346612272) q[6];
ry(-2.178070162980254) q[7];
rz(0.06558370435219886) q[7];
ry(-1.4457976659929197) q[8];
rz(-1.688676914299433) q[8];
ry(-3.1185231855115862) q[9];
rz(-1.6934594167183254) q[9];
ry(-0.09932120766521191) q[10];
rz(2.8814179994427525) q[10];
ry(-0.16030579790366595) q[11];
rz(-1.0906129553510988) q[11];
ry(-1.566606560393386) q[12];
rz(-1.78089400722757) q[12];
ry(-0.7768717953765432) q[13];
rz(3.129944663756791) q[13];
ry(-2.583793012974863) q[14];
rz(-1.069775356601088) q[14];
ry(1.288551360574102) q[15];
rz(-0.00239053436944034) q[15];
ry(-0.0009061166108628171) q[16];
rz(3.1100766747342528) q[16];
ry(0.2429783602789728) q[17];
rz(-3.0330831849442776) q[17];
ry(0.001269473651632354) q[18];
rz(-0.6139584763090369) q[18];
ry(1.752387091000064) q[19];
rz(2.0863715713267936) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.6025359923002531) q[0];
rz(-0.4050858599170804) q[0];
ry(1.6112513487087978) q[1];
rz(-1.5701755657437377) q[1];
ry(3.1029908368912658) q[2];
rz(-1.5390332839034109) q[2];
ry(3.14051165233442) q[3];
rz(-1.4988955241519184) q[3];
ry(-0.0002922687942144009) q[4];
rz(2.5613520410943034) q[4];
ry(0.0015767083181632189) q[5];
rz(-1.5478504458763025) q[5];
ry(2.9364316071257686) q[6];
rz(1.5585351877941431) q[6];
ry(0.00031675767081130573) q[7];
rz(1.5088721426576739) q[7];
ry(3.140931679085942) q[8];
rz(0.560098713681282) q[8];
ry(0.002134317471245595) q[9];
rz(1.9989528548022388) q[9];
ry(-0.0200129642938327) q[10];
rz(2.9492089293353407) q[10];
ry(-0.005079459519327711) q[11];
rz(-0.1941316619820359) q[11];
ry(3.1342436220261374) q[12];
rz(-0.21441499604847095) q[12];
ry(1.5810202379294909) q[13];
rz(-1.5739283330287694) q[13];
ry(-3.1410097292975783) q[14];
rz(-0.959460788174325) q[14];
ry(1.5690979400494247) q[15];
rz(1.5723216263080455) q[15];
ry(3.140359323302735) q[16];
rz(-2.700717952470821) q[16];
ry(-3.0516570990850442) q[17];
rz(1.4398342957579136) q[17];
ry(3.135684428480742) q[18];
rz(0.6178851872160616) q[18];
ry(-2.5989040230198963) q[19];
rz(-0.09857026987232571) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.0018309683513251542) q[0];
rz(-0.09599380469031525) q[0];
ry(1.5703113710232335) q[1];
rz(0.8822666411646708) q[1];
ry(1.5709276671844365) q[2];
rz(-2.0091187564742183) q[2];
ry(-1.5739857346555195) q[3];
rz(-2.0651627818698994) q[3];
ry(-2.511022781891261) q[4];
rz(-0.9423150924909032) q[4];
ry(1.562169328060557) q[5];
rz(-0.1814751612904706) q[5];
ry(-1.5750227764102034) q[6];
rz(-0.8242015758862804) q[6];
ry(1.5728121384672282) q[7];
rz(-0.675151720297671) q[7];
ry(0.19886347797739992) q[8];
rz(1.9593924310847397) q[8];
ry(-1.547082340441661) q[9];
rz(-2.490639919436757) q[9];
ry(-1.4752968006849017) q[10];
rz(-2.035528692210048) q[10];
ry(-1.6181053373654495) q[11];
rz(2.295902197031547) q[11];
ry(-1.5711936241561935) q[12];
rz(-0.48914457272340944) q[12];
ry(1.5737534189914006) q[13];
rz(-1.4211632985979987) q[13];
ry(-1.5542954373163418) q[14];
rz(2.643979505830805) q[14];
ry(1.5704900341868209) q[15];
rz(2.73792805485675) q[15];
ry(-1.5715863248617281) q[16];
rz(-2.064237295874273) q[16];
ry(1.6290762652239028) q[17];
rz(2.7059643826555715) q[17];
ry(1.5741308250953254) q[18];
rz(-2.1534625433844035) q[18];
ry(-0.7660977813944713) q[19];
rz(-2.3716316364500405) q[19];