OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.706519280986418) q[0];
ry(1.9147193677321068) q[1];
cx q[0],q[1];
ry(-1.266173545460698) q[0];
ry(2.0807662674674265) q[1];
cx q[0],q[1];
ry(-0.48156166823766894) q[1];
ry(-1.7063208255271878) q[2];
cx q[1],q[2];
ry(-0.7711083660833999) q[1];
ry(2.2150841689552703) q[2];
cx q[1],q[2];
ry(-1.5899376190350665) q[2];
ry(-0.16783379048839284) q[3];
cx q[2],q[3];
ry(-0.722899047675356) q[2];
ry(-2.653901337235996) q[3];
cx q[2],q[3];
ry(1.5783117604699761) q[3];
ry(0.999336056437394) q[4];
cx q[3],q[4];
ry(-2.787744535719507) q[3];
ry(0.4038528699909536) q[4];
cx q[3],q[4];
ry(2.1735199766362148) q[4];
ry(1.3168460677798324) q[5];
cx q[4],q[5];
ry(-0.4284743317374532) q[4];
ry(-2.9338388839691487) q[5];
cx q[4],q[5];
ry(-1.2212140224903352) q[5];
ry(-1.1890590558944212) q[6];
cx q[5],q[6];
ry(-3.0960336015323464) q[5];
ry(0.30486805364524017) q[6];
cx q[5],q[6];
ry(-0.5925094356988438) q[6];
ry(2.2919969719348945) q[7];
cx q[6],q[7];
ry(-3.113826627275741) q[6];
ry(-0.6003486208457222) q[7];
cx q[6],q[7];
ry(1.6099802220215613) q[7];
ry(-1.5531810718773291) q[8];
cx q[7],q[8];
ry(-1.46154697847954) q[7];
ry(0.07073972984743598) q[8];
cx q[7],q[8];
ry(-3.1141155004430905) q[8];
ry(-2.1823795576674705) q[9];
cx q[8],q[9];
ry(-2.591375981667244) q[8];
ry(-0.33809254642754993) q[9];
cx q[8],q[9];
ry(1.4674807481352588) q[9];
ry(-1.1329300800497302) q[10];
cx q[9],q[10];
ry(-0.023738832943563718) q[9];
ry(-0.0052121505997215975) q[10];
cx q[9],q[10];
ry(0.15055095046464118) q[10];
ry(-1.1687735519950804) q[11];
cx q[10],q[11];
ry(2.373853343986046) q[10];
ry(0.9173742491586284) q[11];
cx q[10],q[11];
ry(2.9222233324594087) q[11];
ry(1.2353026274547023) q[12];
cx q[11],q[12];
ry(-0.4815972422545975) q[11];
ry(-2.6674830086653705) q[12];
cx q[11],q[12];
ry(-0.8763981603413518) q[12];
ry(1.1005098329247245) q[13];
cx q[12],q[13];
ry(1.4049640209248055) q[12];
ry(2.3771074438503117) q[13];
cx q[12],q[13];
ry(2.576668559370622) q[13];
ry(-3.045169407045338) q[14];
cx q[13],q[14];
ry(3.1151001676164203) q[13];
ry(2.007092748482589) q[14];
cx q[13],q[14];
ry(1.040865366856548) q[14];
ry(2.6885868263729935) q[15];
cx q[14],q[15];
ry(-2.1368483083696717) q[14];
ry(0.7853549689770851) q[15];
cx q[14],q[15];
ry(1.6922145355491507) q[15];
ry(-2.9885027484214164) q[16];
cx q[15],q[16];
ry(0.6862740697973334) q[15];
ry(2.545745168388714) q[16];
cx q[15],q[16];
ry(1.3222057478661786) q[16];
ry(-2.5360795463741126) q[17];
cx q[16],q[17];
ry(0.6507283695362123) q[16];
ry(-0.37578956511467426) q[17];
cx q[16],q[17];
ry(-0.4433626940786972) q[17];
ry(-1.5523194752992262) q[18];
cx q[17],q[18];
ry(3.0925690228902516) q[17];
ry(0.4495499355110528) q[18];
cx q[17],q[18];
ry(-2.9730949742629145) q[18];
ry(2.2551420602755554) q[19];
cx q[18],q[19];
ry(-0.18567186697528815) q[18];
ry(-0.7185167573700184) q[19];
cx q[18],q[19];
ry(2.126063147863631) q[0];
ry(1.14824535207975) q[1];
cx q[0],q[1];
ry(-1.6222020557374655) q[0];
ry(1.3244701900057003) q[1];
cx q[0],q[1];
ry(-0.22264093711537747) q[1];
ry(-1.7403901805628768) q[2];
cx q[1],q[2];
ry(-1.1007381601523012) q[1];
ry(2.1336358210285313) q[2];
cx q[1],q[2];
ry(-0.6527945941391193) q[2];
ry(2.895072782350948) q[3];
cx q[2],q[3];
ry(1.8900189191622574) q[2];
ry(2.2293846319176787) q[3];
cx q[2],q[3];
ry(1.6059455811313548) q[3];
ry(-1.9127566230673754) q[4];
cx q[3],q[4];
ry(-1.853506380808802) q[3];
ry(-0.8431149614444426) q[4];
cx q[3],q[4];
ry(2.4440714832777166) q[4];
ry(-2.6259338206972944) q[5];
cx q[4],q[5];
ry(-1.7946654979412617) q[4];
ry(-1.450766746186974) q[5];
cx q[4],q[5];
ry(0.5620395362371964) q[5];
ry(-1.5398464752836745) q[6];
cx q[5],q[6];
ry(3.0868160343502233) q[5];
ry(0.0056298328258040375) q[6];
cx q[5],q[6];
ry(-0.9331727074817602) q[6];
ry(2.004378618687097) q[7];
cx q[6],q[7];
ry(1.0374144379163868) q[6];
ry(0.6312924433126437) q[7];
cx q[6],q[7];
ry(-1.6096442321381312) q[7];
ry(-2.087451102288355) q[8];
cx q[7],q[8];
ry(0.900476362715852) q[7];
ry(0.5799129202266294) q[8];
cx q[7],q[8];
ry(-1.1640565969616654) q[8];
ry(1.9587079024471274) q[9];
cx q[8],q[9];
ry(1.1410762269858676) q[8];
ry(-0.9344441392327534) q[9];
cx q[8],q[9];
ry(-3.063969857634517) q[9];
ry(2.8617870888416848) q[10];
cx q[9],q[10];
ry(0.8112437238261107) q[9];
ry(-2.4356463614334913) q[10];
cx q[9],q[10];
ry(1.043738909178092) q[10];
ry(2.893525670295896) q[11];
cx q[10],q[11];
ry(-0.14969656355215566) q[10];
ry(-1.1827242395072135) q[11];
cx q[10],q[11];
ry(-2.1735722883148965) q[11];
ry(-0.6987944168374723) q[12];
cx q[11],q[12];
ry(3.082303205959088) q[11];
ry(-2.593659313517714) q[12];
cx q[11],q[12];
ry(2.4448833374233434) q[12];
ry(-1.5830063060032797) q[13];
cx q[12],q[13];
ry(-0.2679641853998609) q[12];
ry(-0.015406717216127852) q[13];
cx q[12],q[13];
ry(-0.8789738383601362) q[13];
ry(1.7927957483313413) q[14];
cx q[13],q[14];
ry(0.5214155264245877) q[13];
ry(2.4239127158528992) q[14];
cx q[13],q[14];
ry(-0.75288540973132) q[14];
ry(1.695295192442754) q[15];
cx q[14],q[15];
ry(-2.705194323885693) q[14];
ry(2.7069036649752385) q[15];
cx q[14],q[15];
ry(-1.286270599907304) q[15];
ry(-2.571872829003732) q[16];
cx q[15],q[16];
ry(2.3966621885230146) q[15];
ry(-0.19372909390786222) q[16];
cx q[15],q[16];
ry(-1.1575708221502565) q[16];
ry(-1.1952869868514826) q[17];
cx q[16],q[17];
ry(-1.9512658480040423) q[16];
ry(2.878207858298283) q[17];
cx q[16],q[17];
ry(1.4018775865601452) q[17];
ry(-1.3743975378691031) q[18];
cx q[17],q[18];
ry(-1.9142518532548776) q[17];
ry(0.2511615722971836) q[18];
cx q[17],q[18];
ry(0.3091317417096846) q[18];
ry(2.982825800226785) q[19];
cx q[18],q[19];
ry(2.0203235865795866) q[18];
ry(0.13171138144373096) q[19];
cx q[18],q[19];
ry(-0.3833196343157388) q[0];
ry(2.7255576641344854) q[1];
cx q[0],q[1];
ry(0.8358426905594216) q[0];
ry(-1.4218785895362958) q[1];
cx q[0],q[1];
ry(-1.292260335751488) q[1];
ry(2.3299449158167453) q[2];
cx q[1],q[2];
ry(-2.3383506399388607) q[1];
ry(0.20184377637827566) q[2];
cx q[1],q[2];
ry(1.465074755894669) q[2];
ry(0.3631380065279538) q[3];
cx q[2],q[3];
ry(2.655291375932637) q[2];
ry(0.20687558138916898) q[3];
cx q[2],q[3];
ry(-1.0817999393420106) q[3];
ry(-0.7073495121998213) q[4];
cx q[3],q[4];
ry(0.4337511159424192) q[3];
ry(0.1782407695551866) q[4];
cx q[3],q[4];
ry(-0.6065195873179288) q[4];
ry(-1.4196760926427399) q[5];
cx q[4],q[5];
ry(-2.2085074082377223) q[4];
ry(-0.8135989394119498) q[5];
cx q[4],q[5];
ry(1.4902240358516936) q[5];
ry(-1.8191510314351893) q[6];
cx q[5],q[6];
ry(-3.086833986938485) q[5];
ry(3.107028505378715) q[6];
cx q[5],q[6];
ry(-1.697920717402055) q[6];
ry(-1.775049199579136) q[7];
cx q[6],q[7];
ry(-2.5958975138937697) q[6];
ry(-2.56037728172684) q[7];
cx q[6],q[7];
ry(2.808051332481073) q[7];
ry(-0.2608495541528081) q[8];
cx q[7],q[8];
ry(-2.7677151020898285) q[7];
ry(-2.137160069368883) q[8];
cx q[7],q[8];
ry(-1.7370507085265994) q[8];
ry(0.6101943467560744) q[9];
cx q[8],q[9];
ry(-0.017489805886819525) q[8];
ry(-0.0427160034145837) q[9];
cx q[8],q[9];
ry(2.270050420741542) q[9];
ry(-1.4690900447342914) q[10];
cx q[9],q[10];
ry(-1.6747797556783277) q[9];
ry(-0.344567691067657) q[10];
cx q[9],q[10];
ry(-1.192637908800009) q[10];
ry(3.107844805590166) q[11];
cx q[10],q[11];
ry(-2.228309097810997) q[10];
ry(-0.0355851152684874) q[11];
cx q[10],q[11];
ry(2.833481154571343) q[11];
ry(0.5652655113627301) q[12];
cx q[11],q[12];
ry(3.052368835675798) q[11];
ry(1.6734406337534482) q[12];
cx q[11],q[12];
ry(-0.40087059454225626) q[12];
ry(0.3818999748783999) q[13];
cx q[12],q[13];
ry(1.3948617052856984) q[12];
ry(-3.0294687812615) q[13];
cx q[12],q[13];
ry(-1.7909331552816539) q[13];
ry(-0.6272573563941037) q[14];
cx q[13],q[14];
ry(-0.3260176055099864) q[13];
ry(-0.17722668143106868) q[14];
cx q[13],q[14];
ry(-3.0622137131714786) q[14];
ry(-0.9872199609083693) q[15];
cx q[14],q[15];
ry(3.0239892512303004) q[14];
ry(-1.317679396930986) q[15];
cx q[14],q[15];
ry(2.929746484017244) q[15];
ry(2.148484061626546) q[16];
cx q[15],q[16];
ry(-0.33524000737173343) q[15];
ry(0.012147936001185577) q[16];
cx q[15],q[16];
ry(0.5006757249792839) q[16];
ry(-2.772923288746584) q[17];
cx q[16],q[17];
ry(-0.112267196430766) q[16];
ry(-0.27307343423073327) q[17];
cx q[16],q[17];
ry(-1.3382144263710665) q[17];
ry(2.1702085032307927) q[18];
cx q[17],q[18];
ry(-0.10836515059695745) q[17];
ry(-2.9949871815976326) q[18];
cx q[17],q[18];
ry(2.5564292652400575) q[18];
ry(-1.4347859946769341) q[19];
cx q[18],q[19];
ry(1.2173072757913561) q[18];
ry(-2.481422700690664) q[19];
cx q[18],q[19];
ry(-0.5156269256598023) q[0];
ry(1.102650125145117) q[1];
cx q[0],q[1];
ry(1.921381643556603) q[0];
ry(-2.2506171881286496) q[1];
cx q[0],q[1];
ry(0.391915915394927) q[1];
ry(-0.5969524300084474) q[2];
cx q[1],q[2];
ry(-2.6697192446181903) q[1];
ry(2.2684011095394787) q[2];
cx q[1],q[2];
ry(-2.0155379774009576) q[2];
ry(2.7733735438030753) q[3];
cx q[2],q[3];
ry(1.244363094551783) q[2];
ry(1.3063584888600945) q[3];
cx q[2],q[3];
ry(2.4803193877251206) q[3];
ry(2.560014314528241) q[4];
cx q[3],q[4];
ry(-1.5601537087936146) q[3];
ry(-1.4794931489370782) q[4];
cx q[3],q[4];
ry(-1.7757990793372305) q[4];
ry(1.9180254092987534) q[5];
cx q[4],q[5];
ry(0.008968679175409555) q[4];
ry(1.3032361102133914) q[5];
cx q[4],q[5];
ry(0.9314682641517207) q[5];
ry(2.268279441701454) q[6];
cx q[5],q[6];
ry(-3.0882424754883107) q[5];
ry(0.012274462358626614) q[6];
cx q[5],q[6];
ry(-2.7629825200198135) q[6];
ry(2.630341890448173) q[7];
cx q[6],q[7];
ry(0.31801631950547105) q[6];
ry(-2.7104564524735224) q[7];
cx q[6],q[7];
ry(1.4337058597320482) q[7];
ry(-0.11935396834853038) q[8];
cx q[7],q[8];
ry(0.6653994859877638) q[7];
ry(-0.7315729911295918) q[8];
cx q[7],q[8];
ry(-1.3146736632677172) q[8];
ry(-1.769836050011464) q[9];
cx q[8],q[9];
ry(3.076178409959037) q[8];
ry(-2.9461538114137347) q[9];
cx q[8],q[9];
ry(0.2612107265738297) q[9];
ry(-0.16491596200633565) q[10];
cx q[9],q[10];
ry(-0.30687055210863806) q[9];
ry(-1.860757816041029) q[10];
cx q[9],q[10];
ry(-1.545553998443518) q[10];
ry(1.75121811274097) q[11];
cx q[10],q[11];
ry(-0.7853381180462344) q[10];
ry(-0.09362263875403354) q[11];
cx q[10],q[11];
ry(0.14774055667438626) q[11];
ry(0.13302243690695903) q[12];
cx q[11],q[12];
ry(-2.7449298137262383) q[11];
ry(-0.04651815499497359) q[12];
cx q[11],q[12];
ry(2.05563929017571) q[12];
ry(-1.9575036419608107) q[13];
cx q[12],q[13];
ry(3.053795947036627) q[12];
ry(-0.043839556845003534) q[13];
cx q[12],q[13];
ry(3.0650614032408234) q[13];
ry(-0.7700890780453831) q[14];
cx q[13],q[14];
ry(0.05326985231636464) q[13];
ry(3.1364785494568994) q[14];
cx q[13],q[14];
ry(-2.06096139239614) q[14];
ry(-1.1147526810142043) q[15];
cx q[14],q[15];
ry(0.012768681132286587) q[14];
ry(1.2773149413871572) q[15];
cx q[14],q[15];
ry(2.8470094205120433) q[15];
ry(2.443532717310734) q[16];
cx q[15],q[16];
ry(-0.5703606784177246) q[15];
ry(-0.5470111374131107) q[16];
cx q[15],q[16];
ry(0.7769016527360852) q[16];
ry(2.8769467995168525) q[17];
cx q[16],q[17];
ry(-1.657641296768067) q[16];
ry(-2.5720403373332372) q[17];
cx q[16],q[17];
ry(1.4389211916704872) q[17];
ry(-0.6658809451882846) q[18];
cx q[17],q[18];
ry(-0.3081865111501072) q[17];
ry(1.0815762655259213) q[18];
cx q[17],q[18];
ry(0.21289847624948077) q[18];
ry(0.9115508038509974) q[19];
cx q[18],q[19];
ry(1.2911180598680332) q[18];
ry(1.3559161600653535) q[19];
cx q[18],q[19];
ry(0.6672095444480042) q[0];
ry(-1.9941986123283133) q[1];
cx q[0],q[1];
ry(-2.4747400228141556) q[0];
ry(-2.5413403606608314) q[1];
cx q[0],q[1];
ry(1.3225346717047772) q[1];
ry(-2.7131687911825475) q[2];
cx q[1],q[2];
ry(0.7503490487111604) q[1];
ry(2.22319264133895) q[2];
cx q[1],q[2];
ry(0.993273138991916) q[2];
ry(-1.1765883121182634) q[3];
cx q[2],q[3];
ry(-1.8251529192802431) q[2];
ry(0.8964418966884251) q[3];
cx q[2],q[3];
ry(-2.133994827518912) q[3];
ry(2.202101500273246) q[4];
cx q[3],q[4];
ry(-1.3445024651112876) q[3];
ry(1.444350215766061) q[4];
cx q[3],q[4];
ry(1.7667308695728847) q[4];
ry(-0.7887022452536337) q[5];
cx q[4],q[5];
ry(2.0898251327518667) q[4];
ry(2.793761626947135) q[5];
cx q[4],q[5];
ry(-1.4131649820748828) q[5];
ry(2.566229647192817) q[6];
cx q[5],q[6];
ry(-0.05383714939347062) q[5];
ry(-0.005363696024442402) q[6];
cx q[5],q[6];
ry(1.605260362514137) q[6];
ry(1.4388623153802562) q[7];
cx q[6],q[7];
ry(1.9890033888958651) q[6];
ry(-1.702747912589226) q[7];
cx q[6],q[7];
ry(-0.7243848108500437) q[7];
ry(-0.8770896833524162) q[8];
cx q[7],q[8];
ry(0.1552400974338042) q[7];
ry(-0.13625482484481882) q[8];
cx q[7],q[8];
ry(-0.9954834388767707) q[8];
ry(-2.18467860072768) q[9];
cx q[8],q[9];
ry(1.6320321885180258) q[8];
ry(-2.9420237514470355) q[9];
cx q[8],q[9];
ry(1.0429364236057326) q[9];
ry(-0.8179519484078632) q[10];
cx q[9],q[10];
ry(0.009244585626093212) q[9];
ry(-3.0891530421116045) q[10];
cx q[9],q[10];
ry(-0.7802475944660241) q[10];
ry(2.5542367506281582) q[11];
cx q[10],q[11];
ry(3.1000712891675386) q[10];
ry(-1.0846903074112308) q[11];
cx q[10],q[11];
ry(1.6350916693215218) q[11];
ry(-0.4096956945917922) q[12];
cx q[11],q[12];
ry(-0.42643677362412014) q[11];
ry(1.6159359648880605) q[12];
cx q[11],q[12];
ry(2.3259626056889826) q[12];
ry(-2.557682219910261) q[13];
cx q[12],q[13];
ry(-1.4671513969287782) q[12];
ry(1.220398056563767) q[13];
cx q[12],q[13];
ry(-1.641477402839612) q[13];
ry(-3.13989218405395) q[14];
cx q[13],q[14];
ry(-1.7858461339089664) q[13];
ry(2.359573362608482) q[14];
cx q[13],q[14];
ry(-1.5527097227988944) q[14];
ry(0.5927348204877927) q[15];
cx q[14],q[15];
ry(1.6626467560646647) q[14];
ry(-1.7292592250527974) q[15];
cx q[14],q[15];
ry(1.5250104816989947) q[15];
ry(-1.5710754114492183) q[16];
cx q[15],q[16];
ry(1.7080450559101248) q[15];
ry(2.1321959196074918) q[16];
cx q[15],q[16];
ry(1.5524742077125626) q[16];
ry(-1.6838324361310635) q[17];
cx q[16],q[17];
ry(-1.975634541252555) q[16];
ry(-2.419813131203383) q[17];
cx q[16],q[17];
ry(-2.8862198162451285) q[17];
ry(-2.463153376846658) q[18];
cx q[17],q[18];
ry(3.0752758101913154) q[17];
ry(0.024815405634432963) q[18];
cx q[17],q[18];
ry(0.005333057490378211) q[18];
ry(-2.466504750774584) q[19];
cx q[18],q[19];
ry(2.116340958249448) q[18];
ry(-2.8920276224894597) q[19];
cx q[18],q[19];
ry(-2.021725326395715) q[0];
ry(-2.1040484533705337) q[1];
cx q[0],q[1];
ry(1.439635024736675) q[0];
ry(-2.651568128604909) q[1];
cx q[0],q[1];
ry(1.0593973703410007) q[1];
ry(-1.2455280115344296) q[2];
cx q[1],q[2];
ry(-1.957258642248375) q[1];
ry(-1.4328679452234367) q[2];
cx q[1],q[2];
ry(-1.8396187781866273) q[2];
ry(-0.6348158195994291) q[3];
cx q[2],q[3];
ry(-1.3459543867713517) q[2];
ry(0.17871519714541467) q[3];
cx q[2],q[3];
ry(0.9995930325157564) q[3];
ry(0.3793928966804337) q[4];
cx q[3],q[4];
ry(1.8214394029272594) q[3];
ry(1.4348545414389378) q[4];
cx q[3],q[4];
ry(0.4507543343494957) q[4];
ry(-0.9463333445832971) q[5];
cx q[4],q[5];
ry(2.9752007724522262) q[4];
ry(0.942628803575615) q[5];
cx q[4],q[5];
ry(-1.1900737246430015) q[5];
ry(1.2294594439808275) q[6];
cx q[5],q[6];
ry(1.2909375106436576) q[5];
ry(-0.03344851829842188) q[6];
cx q[5],q[6];
ry(1.8418498051197283) q[6];
ry(0.3629856651089078) q[7];
cx q[6],q[7];
ry(2.8969250989077975) q[6];
ry(2.6625055423155737) q[7];
cx q[6],q[7];
ry(-2.3205569173882257) q[7];
ry(-0.6219177410117911) q[8];
cx q[7],q[8];
ry(-1.3492543602214502) q[7];
ry(3.1159744220444945) q[8];
cx q[7],q[8];
ry(1.5687857108695267) q[8];
ry(-2.281519007016766) q[9];
cx q[8],q[9];
ry(3.0347605282298327) q[8];
ry(1.973887181150667) q[9];
cx q[8],q[9];
ry(1.349665674408567) q[9];
ry(1.2390713776058613) q[10];
cx q[9],q[10];
ry(-0.44388861014096115) q[9];
ry(0.839924064784781) q[10];
cx q[9],q[10];
ry(-0.39880637105510613) q[10];
ry(-1.4401359110072893) q[11];
cx q[10],q[11];
ry(-3.125688124836354) q[10];
ry(3.1381006066163777) q[11];
cx q[10],q[11];
ry(0.7598547837516394) q[11];
ry(2.5188745073371774) q[12];
cx q[11],q[12];
ry(3.1317571472401178) q[11];
ry(3.112330047524884) q[12];
cx q[11],q[12];
ry(0.5354987747423771) q[12];
ry(1.6337278937759674) q[13];
cx q[12],q[13];
ry(0.4344187031972276) q[12];
ry(0.05118613178832376) q[13];
cx q[12],q[13];
ry(2.2720381708826305) q[13];
ry(1.9118895797083901) q[14];
cx q[13],q[14];
ry(3.0703629854664185) q[13];
ry(-3.1312903363236924) q[14];
cx q[13],q[14];
ry(-1.2741891029846588) q[14];
ry(-2.993196520658775) q[15];
cx q[14],q[15];
ry(3.135496110356327) q[14];
ry(0.40325743311167167) q[15];
cx q[14],q[15];
ry(0.13432947674933615) q[15];
ry(0.37025651977801743) q[16];
cx q[15],q[16];
ry(3.133347278065163) q[15];
ry(0.9948154380289754) q[16];
cx q[15],q[16];
ry(0.3630355610987239) q[16];
ry(0.4055776190310559) q[17];
cx q[16],q[17];
ry(1.7481146235274811) q[16];
ry(1.270667930650256) q[17];
cx q[16],q[17];
ry(-3.0265786816662725) q[17];
ry(-1.0245864790034673) q[18];
cx q[17],q[18];
ry(2.83643254111271) q[17];
ry(3.121232783335348) q[18];
cx q[17],q[18];
ry(1.8536815913278517) q[18];
ry(2.851627632413794) q[19];
cx q[18],q[19];
ry(1.3494383802800964) q[18];
ry(1.8742981120210531) q[19];
cx q[18],q[19];
ry(0.5139983886037909) q[0];
ry(2.6566035702745827) q[1];
cx q[0],q[1];
ry(2.619734239572281) q[0];
ry(1.7890615217993897) q[1];
cx q[0],q[1];
ry(1.7834470647450722) q[1];
ry(-0.8866045720991251) q[2];
cx q[1],q[2];
ry(1.42958664518168) q[1];
ry(1.9957074298194095) q[2];
cx q[1],q[2];
ry(2.772796458113401) q[2];
ry(-0.5083758751336038) q[3];
cx q[2],q[3];
ry(3.119300204064497) q[2];
ry(2.5187662039119645) q[3];
cx q[2],q[3];
ry(0.22012300352937242) q[3];
ry(-0.3141346157672395) q[4];
cx q[3],q[4];
ry(-1.087676085610246) q[3];
ry(-2.4441568412867816) q[4];
cx q[3],q[4];
ry(-2.952937927509548) q[4];
ry(2.028495353880394) q[5];
cx q[4],q[5];
ry(-1.3761149886310573) q[4];
ry(-0.1523438654474722) q[5];
cx q[4],q[5];
ry(2.2810430062360103) q[5];
ry(0.6259468394073187) q[6];
cx q[5],q[6];
ry(3.1145096052597525) q[5];
ry(-0.04251869408666131) q[6];
cx q[5],q[6];
ry(1.6808578609156724) q[6];
ry(0.962529362727178) q[7];
cx q[6],q[7];
ry(-3.1290536604440016) q[6];
ry(0.12454019203793118) q[7];
cx q[6],q[7];
ry(-1.550520373718025) q[7];
ry(2.1216751423061173) q[8];
cx q[7],q[8];
ry(2.842871787831204) q[7];
ry(1.6268247712852082) q[8];
cx q[7],q[8];
ry(-1.0991008336484824) q[8];
ry(0.19202144697356965) q[9];
cx q[8],q[9];
ry(0.0007308000175463292) q[8];
ry(-3.1185465057766004) q[9];
cx q[8],q[9];
ry(1.0160151678340927) q[9];
ry(-0.32571493253412576) q[10];
cx q[9],q[10];
ry(2.009744224241349) q[9];
ry(-3.115473197060857) q[10];
cx q[9],q[10];
ry(1.598176964593703) q[10];
ry(-2.7100202449132134) q[11];
cx q[10],q[11];
ry(1.5344670904691442) q[10];
ry(-1.1984070358807424) q[11];
cx q[10],q[11];
ry(1.572748029454025) q[11];
ry(-0.6473347448050069) q[12];
cx q[11],q[12];
ry(3.1407765293838397) q[11];
ry(0.9712922159802142) q[12];
cx q[11],q[12];
ry(-2.6983278787534783) q[12];
ry(-2.1434351086274086) q[13];
cx q[12],q[13];
ry(3.1107323806798384) q[12];
ry(-1.404136891367) q[13];
cx q[12],q[13];
ry(-2.885779767070884) q[13];
ry(1.5639245226295984) q[14];
cx q[13],q[14];
ry(-3.054272827758053) q[13];
ry(1.7217466972458284) q[14];
cx q[13],q[14];
ry(-1.5872554444747986) q[14];
ry(-1.546240726405645) q[15];
cx q[14],q[15];
ry(2.6238564245185767) q[14];
ry(-1.2242937418231836) q[15];
cx q[14],q[15];
ry(-1.6088539273425706) q[15];
ry(2.4541004676023848) q[16];
cx q[15],q[16];
ry(3.1414152497058607) q[15];
ry(1.6894423427943508) q[16];
cx q[15],q[16];
ry(-0.7454203481449762) q[16];
ry(0.480132264983828) q[17];
cx q[16],q[17];
ry(-2.001191435030787) q[16];
ry(-1.3792166816847793) q[17];
cx q[16],q[17];
ry(-0.8857267478012901) q[17];
ry(-2.8153291071082354) q[18];
cx q[17],q[18];
ry(0.5816861942912039) q[17];
ry(2.5521751086175852) q[18];
cx q[17],q[18];
ry(-1.5586797602045257) q[18];
ry(0.03332876343943757) q[19];
cx q[18],q[19];
ry(-2.276044363835148) q[18];
ry(-1.8130708799325557) q[19];
cx q[18],q[19];
ry(2.777695483863943) q[0];
ry(0.6785698679482239) q[1];
cx q[0],q[1];
ry(0.6880240472987468) q[0];
ry(1.0709288238115269) q[1];
cx q[0],q[1];
ry(-2.3269022317621317) q[1];
ry(3.0052444076365323) q[2];
cx q[1],q[2];
ry(1.8918387631029416) q[1];
ry(1.4794912509018079) q[2];
cx q[1],q[2];
ry(1.8190199898002712) q[2];
ry(1.9562635863199669) q[3];
cx q[2],q[3];
ry(0.6932902290268896) q[2];
ry(-2.3467920761633643) q[3];
cx q[2],q[3];
ry(-2.9375446454301133) q[3];
ry(0.3354461299628186) q[4];
cx q[3],q[4];
ry(-3.131666548653464) q[3];
ry(1.6838522964976224) q[4];
cx q[3],q[4];
ry(-0.8335965154413854) q[4];
ry(-1.3228845330889396) q[5];
cx q[4],q[5];
ry(-2.8247754171116886) q[4];
ry(2.7391530905866848) q[5];
cx q[4],q[5];
ry(2.6055374259757356) q[5];
ry(-0.9336590560860145) q[6];
cx q[5],q[6];
ry(-3.11131821875457) q[5];
ry(1.5301750948538686) q[6];
cx q[5],q[6];
ry(0.37362200880408825) q[6];
ry(1.5894253655668873) q[7];
cx q[6],q[7];
ry(-2.228605039715803) q[6];
ry(-2.2451519582379205) q[7];
cx q[6],q[7];
ry(-1.5640811670546142) q[7];
ry(-1.700726837128168) q[8];
cx q[7],q[8];
ry(-2.60956529192292) q[7];
ry(1.4695334684057029) q[8];
cx q[7],q[8];
ry(1.5709360257700655) q[8];
ry(2.7574608948084953) q[9];
cx q[8],q[9];
ry(-1.6320634455444556) q[8];
ry(-1.5904648092132143) q[9];
cx q[8],q[9];
ry(-1.567957280952431) q[9];
ry(1.5802430175450466) q[10];
cx q[9],q[10];
ry(-0.40484593435403093) q[9];
ry(-0.7329456184473532) q[10];
cx q[9],q[10];
ry(-1.5792585649733564) q[10];
ry(-3.0534772476758723) q[11];
cx q[10],q[11];
ry(-0.0034363928711709235) q[10];
ry(-1.96588096853196) q[11];
cx q[10],q[11];
ry(3.055155554804502) q[11];
ry(-1.565941595574625) q[12];
cx q[11],q[12];
ry(-1.5693693757229588) q[11];
ry(-1.9040468156316415) q[12];
cx q[11],q[12];
ry(1.5688422290158917) q[12];
ry(1.5707207665619538) q[13];
cx q[12],q[13];
ry(1.9146797671142668) q[12];
ry(2.2867079663485415) q[13];
cx q[12],q[13];
ry(1.5630424279183857) q[13];
ry(1.5672731688138857) q[14];
cx q[13],q[14];
ry(1.6649893973627368) q[13];
ry(-3.0112398318232394) q[14];
cx q[13],q[14];
ry(-1.5705668492786322) q[14];
ry(-1.575987634313276) q[15];
cx q[14],q[15];
ry(-0.8233441496959948) q[14];
ry(1.6539794213406036) q[15];
cx q[14],q[15];
ry(-0.9861736643240975) q[15];
ry(1.578714811412759) q[16];
cx q[15],q[16];
ry(1.3104866497289505) q[15];
ry(-3.1155029938342014) q[16];
cx q[15],q[16];
ry(-1.5589331988172304) q[16];
ry(1.5781499842389446) q[17];
cx q[16],q[17];
ry(1.1965122402617219) q[16];
ry(-1.6950214269166843) q[17];
cx q[16],q[17];
ry(-1.487954824332064) q[17];
ry(-2.000159803184338) q[18];
cx q[17],q[18];
ry(0.24211080690422548) q[17];
ry(1.4065236220329869) q[18];
cx q[17],q[18];
ry(0.20027857943859814) q[18];
ry(2.839145476063103) q[19];
cx q[18],q[19];
ry(2.7749312443710235) q[18];
ry(-0.1601161828773813) q[19];
cx q[18],q[19];
ry(0.01683728531676112) q[0];
ry(0.8102684359001711) q[1];
cx q[0],q[1];
ry(-2.2905667327100474) q[0];
ry(0.8734761804250599) q[1];
cx q[0],q[1];
ry(1.8910941930130285) q[1];
ry(2.521787943623691) q[2];
cx q[1],q[2];
ry(-2.309218185031801) q[1];
ry(0.4088018396577855) q[2];
cx q[1],q[2];
ry(-0.6979186381183213) q[2];
ry(-1.209092535178712) q[3];
cx q[2],q[3];
ry(-0.10841335840719779) q[2];
ry(0.8617101776367626) q[3];
cx q[2],q[3];
ry(-1.7319976565283257) q[3];
ry(0.5327260719396198) q[4];
cx q[3],q[4];
ry(1.080971349082787) q[3];
ry(-0.10549178224224795) q[4];
cx q[3],q[4];
ry(1.5636301940612) q[4];
ry(-1.602836952369284) q[5];
cx q[4],q[5];
ry(2.2816690925592007) q[4];
ry(1.579238210489586) q[5];
cx q[4],q[5];
ry(-1.5699390844969972) q[5];
ry(-1.568988185765536) q[6];
cx q[5],q[6];
ry(2.9783423082595375) q[5];
ry(1.9032760146473875) q[6];
cx q[5],q[6];
ry(-1.5679571155884764) q[6];
ry(0.6462330608742817) q[7];
cx q[6],q[7];
ry(0.0005035196880109893) q[6];
ry(-1.3488512593103523) q[7];
cx q[6],q[7];
ry(-0.646736620279957) q[7];
ry(2.771075370446863) q[8];
cx q[7],q[8];
ry(0.00012085240727266466) q[7];
ry(2.028605817137898) q[8];
cx q[7],q[8];
ry(-2.7688700024411146) q[8];
ry(-1.7903959786825416) q[9];
cx q[8],q[9];
ry(-3.1225167064652037) q[8];
ry(-1.806564833596937) q[9];
cx q[8],q[9];
ry(1.3541302501212185) q[9];
ry(1.5716702082028198) q[10];
cx q[9],q[10];
ry(-1.9921158757239308) q[9];
ry(-2.3394992163638966) q[10];
cx q[9],q[10];
ry(1.56936744002985) q[10];
ry(-1.5707609538681995) q[11];
cx q[10],q[11];
ry(-1.9160764641130328) q[10];
ry(1.9550525041921958) q[11];
cx q[10],q[11];
ry(1.5723382993258115) q[11];
ry(-2.9968888460496577) q[12];
cx q[11],q[12];
ry(0.0032248981530402783) q[11];
ry(2.232125877247039) q[12];
cx q[11],q[12];
ry(-0.14426876522822174) q[12];
ry(1.564069320731499) q[13];
cx q[12],q[13];
ry(-1.2934992495958781) q[12];
ry(1.2702389946530988) q[13];
cx q[12],q[13];
ry(-1.572407256521629) q[13];
ry(-1.5829377792155208) q[14];
cx q[13],q[14];
ry(-2.641330535003472) q[13];
ry(-1.1248265656004204) q[14];
cx q[13],q[14];
ry(-1.5783057498234987) q[14];
ry(2.163103163148222) q[15];
cx q[14],q[15];
ry(-1.8068743804068088) q[14];
ry(0.8794926905937421) q[15];
cx q[14],q[15];
ry(1.7575348810491453) q[15];
ry(-0.28394317853260187) q[16];
cx q[15],q[16];
ry(-2.6915067907929657) q[15];
ry(-2.1485457016867495) q[16];
cx q[15],q[16];
ry(-0.5187623148181845) q[16];
ry(1.505931428248373) q[17];
cx q[16],q[17];
ry(-3.1071736467265643) q[16];
ry(-0.0017748248130317816) q[17];
cx q[16],q[17];
ry(0.6942444160234498) q[17];
ry(1.4865926274520938) q[18];
cx q[17],q[18];
ry(0.2030805412570036) q[17];
ry(-3.0371380306191624) q[18];
cx q[17],q[18];
ry(-1.2936306384852085) q[18];
ry(0.19392183324270018) q[19];
cx q[18],q[19];
ry(2.8546085610771694) q[18];
ry(-0.2343777133474544) q[19];
cx q[18],q[19];
ry(2.3896777027093385) q[0];
ry(1.8118207566212137) q[1];
cx q[0],q[1];
ry(1.574850954443991) q[0];
ry(1.7823180751165804) q[1];
cx q[0],q[1];
ry(-1.812680049860201) q[1];
ry(2.3818296558913272) q[2];
cx q[1],q[2];
ry(0.5794785034889546) q[1];
ry(-1.558820758858361) q[2];
cx q[1],q[2];
ry(-0.5725207878268872) q[2];
ry(-0.3347529531354673) q[3];
cx q[2],q[3];
ry(-0.5090752542515631) q[2];
ry(-1.6920300731146134) q[3];
cx q[2],q[3];
ry(0.8350356319511374) q[3];
ry(1.5943732317897403) q[4];
cx q[3],q[4];
ry(1.4782879078042237) q[3];
ry(0.0020152748101223135) q[4];
cx q[3],q[4];
ry(-2.7180145475121593) q[4];
ry(-1.5782628476956557) q[5];
cx q[4],q[5];
ry(2.4359556195975256) q[4];
ry(0.0011439926472854567) q[5];
cx q[4],q[5];
ry(-0.8006880878943771) q[5];
ry(1.5719019933494005) q[6];
cx q[5],q[6];
ry(1.9550818210320333) q[5];
ry(-0.0012468199407006705) q[6];
cx q[5],q[6];
ry(2.2389685068493232) q[6];
ry(-1.5710826939336786) q[7];
cx q[6],q[7];
ry(-2.0523245083805026) q[6];
ry(-0.0012241361875506842) q[7];
cx q[6],q[7];
ry(1.5700257009241856) q[7];
ry(-2.4269463282024293) q[8];
cx q[7],q[8];
ry(0.003926348757296524) q[7];
ry(1.7901027299256624) q[8];
cx q[7],q[8];
ry(-0.7141712431823796) q[8];
ry(-1.5685429287474584) q[9];
cx q[8],q[9];
ry(-1.3794772205350068) q[8];
ry(1.9458196578236673) q[9];
cx q[8],q[9];
ry(-0.6438815710440419) q[9];
ry(-1.5719455565522686) q[10];
cx q[9],q[10];
ry(-0.30453553006829737) q[9];
ry(3.1414146474659788) q[10];
cx q[9],q[10];
ry(-1.5336883668303818) q[10];
ry(1.84399707002031) q[11];
cx q[10],q[11];
ry(3.1404746339548995) q[10];
ry(3.020396636610998) q[11];
cx q[10],q[11];
ry(-1.299532598041277) q[11];
ry(-1.5713185030573389) q[12];
cx q[11],q[12];
ry(1.1718205625408054) q[11];
ry(-1.4639917210266942) q[12];
cx q[11],q[12];
ry(-1.5717680372265335) q[12];
ry(-1.5710050823205388) q[13];
cx q[12],q[13];
ry(2.2695934335532084) q[12];
ry(1.9290889313452113) q[13];
cx q[12],q[13];
ry(-0.026216974410105287) q[13];
ry(-0.029661848727456253) q[14];
cx q[13],q[14];
ry(1.5183559706465062) q[13];
ry(-1.5463415093230244) q[14];
cx q[13],q[14];
ry(1.972154009870548) q[14];
ry(3.13739075692595) q[15];
cx q[14],q[15];
ry(3.1245155051300983) q[14];
ry(-0.013728448575633045) q[15];
cx q[14],q[15];
ry(-0.3611718960986696) q[15];
ry(-1.1266052553494101) q[16];
cx q[15],q[16];
ry(-1.1390038482392324) q[15];
ry(-2.575989157406378) q[16];
cx q[15],q[16];
ry(1.0802051913736408) q[16];
ry(-2.3852964514705683) q[17];
cx q[16],q[17];
ry(-0.9426898775432938) q[16];
ry(-1.0048118198056109) q[17];
cx q[16],q[17];
ry(-1.524176877314881) q[17];
ry(-0.007687564984218097) q[18];
cx q[17],q[18];
ry(1.7102828357311015) q[17];
ry(-0.07684977179814735) q[18];
cx q[17],q[18];
ry(0.829643667289582) q[18];
ry(1.6176165483894829) q[19];
cx q[18],q[19];
ry(1.5503728835568424) q[18];
ry(-0.48248951766585313) q[19];
cx q[18],q[19];
ry(-1.2570512865196912) q[0];
ry(1.9734576638122459) q[1];
cx q[0],q[1];
ry(-1.7998088532748202) q[0];
ry(-1.649291805644144) q[1];
cx q[0],q[1];
ry(1.85911479033853) q[1];
ry(0.47987742018635426) q[2];
cx q[1],q[2];
ry(3.1359785801926403) q[1];
ry(-1.7585685793822066) q[2];
cx q[1],q[2];
ry(-0.18721500397449906) q[2];
ry(0.4747879670536497) q[3];
cx q[2],q[3];
ry(-0.8192960375285859) q[2];
ry(0.42729592271218014) q[3];
cx q[2],q[3];
ry(-0.004922031038091745) q[3];
ry(2.7109831972581677) q[4];
cx q[3],q[4];
ry(-1.678354519260326) q[3];
ry(2.4902911333330287) q[4];
cx q[3],q[4];
ry(1.5772258426658106) q[4];
ry(0.7936037747428353) q[5];
cx q[4],q[5];
ry(1.70066340036745) q[4];
ry(1.572452084258017) q[5];
cx q[4],q[5];
ry(-2.818177174594749) q[5];
ry(-2.239662836256813) q[6];
cx q[5],q[6];
ry(-1.4381948538338252) q[5];
ry(-0.0004669881647663132) q[6];
cx q[5],q[6];
ry(1.57239378722151) q[6];
ry(1.5330275749926559) q[7];
cx q[6],q[7];
ry(0.12215739232777632) q[6];
ry(-1.5246846321981284) q[7];
cx q[6],q[7];
ry(-1.5312593011989417) q[7];
ry(1.5705722823033943) q[8];
cx q[7],q[8];
ry(1.6981797979912703) q[7];
ry(-2.7939084941512595) q[8];
cx q[7],q[8];
ry(1.5705646069255568) q[8];
ry(-0.6415484269315048) q[9];
cx q[8],q[9];
ry(1.1228251168746957) q[8];
ry(-1.7974091342799827) q[9];
cx q[8],q[9];
ry(-2.956970850591735) q[9];
ry(-1.5349366991366715) q[10];
cx q[9],q[10];
ry(0.9523005719615059) q[9];
ry(-3.1362253596909846) q[10];
cx q[9],q[10];
ry(-0.48778191277572175) q[10];
ry(-1.5710846465844448) q[11];
cx q[10],q[11];
ry(0.9931025640720623) q[10];
ry(-3.139883068764571) q[11];
cx q[10],q[11];
ry(1.5702394628177248) q[11];
ry(-1.5400627196169416) q[12];
cx q[11],q[12];
ry(-1.877121706056669) q[11];
ry(1.0749119592644982) q[12];
cx q[11],q[12];
ry(0.02603630893612774) q[12];
ry(-0.5243310205707016) q[13];
cx q[12],q[13];
ry(-0.022189009885649914) q[12];
ry(3.112858196801139) q[13];
cx q[12],q[13];
ry(-0.6870789513757304) q[13];
ry(3.0066121670213777) q[14];
cx q[13],q[14];
ry(1.587938202503702) q[13];
ry(-1.5861329602520424) q[14];
cx q[13],q[14];
ry(0.05952548505307456) q[14];
ry(1.2707189369030512) q[15];
cx q[14],q[15];
ry(1.818976244927101) q[14];
ry(-1.8428500149465) q[15];
cx q[14],q[15];
ry(1.570286297876822) q[15];
ry(-1.5691700151359718) q[16];
cx q[15],q[16];
ry(0.6930445919964816) q[15];
ry(-0.8076617554319983) q[16];
cx q[15],q[16];
ry(-2.319584711764205) q[16];
ry(-1.629086347720124) q[17];
cx q[16],q[17];
ry(-1.9493325455966986) q[16];
ry(-0.0042334691734244295) q[17];
cx q[16],q[17];
ry(-2.0157900522698187) q[17];
ry(1.7704252008576924) q[18];
cx q[17],q[18];
ry(-3.135784566465081) q[17];
ry(-0.0018836041371530499) q[18];
cx q[17],q[18];
ry(-0.3907829829349838) q[18];
ry(-0.4217857803890954) q[19];
cx q[18],q[19];
ry(2.9704260434888345) q[18];
ry(-1.345707579080594) q[19];
cx q[18],q[19];
ry(-1.0864434061035162) q[0];
ry(-2.2054253594532747) q[1];
cx q[0],q[1];
ry(-0.8450847782572417) q[0];
ry(1.5979375443302324) q[1];
cx q[0],q[1];
ry(2.447185140390105) q[1];
ry(2.3766523858144497) q[2];
cx q[1],q[2];
ry(0.07722156539998881) q[1];
ry(-0.2351121160698553) q[2];
cx q[1],q[2];
ry(1.6118550042999402) q[2];
ry(-1.5829884016552684) q[3];
cx q[2],q[3];
ry(-2.179424594191255) q[2];
ry(1.5966092060698405) q[3];
cx q[2],q[3];
ry(-1.5838062932192631) q[3];
ry(1.5728853650110466) q[4];
cx q[3],q[4];
ry(-1.327144194278636) q[3];
ry(1.5427516661328085) q[4];
cx q[3],q[4];
ry(1.5719900314993538) q[4];
ry(2.840206878470317) q[5];
cx q[4],q[5];
ry(-3.0335571340236447) q[4];
ry(-1.5315833288705811) q[5];
cx q[4],q[5];
ry(-1.5471988721654002) q[5];
ry(-1.5747128603865632) q[6];
cx q[5],q[6];
ry(-0.12999764636454242) q[5];
ry(-1.785516822885517) q[6];
cx q[5],q[6];
ry(1.5646546255612745) q[6];
ry(-1.5854802118878606) q[7];
cx q[6],q[7];
ry(0.1629193593816115) q[6];
ry(3.0393155917251953) q[7];
cx q[6],q[7];
ry(0.2529733674582105) q[7];
ry(1.5713554722306142) q[8];
cx q[7],q[8];
ry(1.613766813963771) q[7];
ry(3.1412871023090605) q[8];
cx q[7],q[8];
ry(2.847276771904291) q[8];
ry(-2.9677191156386877) q[9];
cx q[8],q[9];
ry(-0.12668317306481924) q[8];
ry(0.0022859597671703786) q[9];
cx q[8],q[9];
ry(1.5831821372878774) q[9];
ry(-0.4877440782349017) q[10];
cx q[9],q[10];
ry(-1.047638081144498) q[9];
ry(-1.8871284537068345) q[10];
cx q[9],q[10];
ry(1.5844433302004042) q[10];
ry(1.5561356101697497) q[11];
cx q[10],q[11];
ry(3.1333519910010743) q[10];
ry(-3.0247964112106773) q[11];
cx q[10],q[11];
ry(1.5561524848961237) q[11];
ry(-0.0163517669117752) q[12];
cx q[11],q[12];
ry(0.7532277179673743) q[11];
ry(0.802626078292153) q[12];
cx q[11],q[12];
ry(-2.6445286630677596) q[12];
ry(0.03197192355390988) q[13];
cx q[12],q[13];
ry(-1.986481772003433) q[12];
ry(3.1410763033507783) q[13];
cx q[12],q[13];
ry(-1.571278569008612) q[13];
ry(1.5714452443431188) q[14];
cx q[13],q[14];
ry(1.4496289021019075) q[13];
ry(1.5056058241658878) q[14];
cx q[13],q[14];
ry(1.5702229012240931) q[14];
ry(-1.5694466062134422) q[15];
cx q[14],q[15];
ry(-1.1474410063620863) q[14];
ry(-0.8268443912331928) q[15];
cx q[14],q[15];
ry(0.30899245171230477) q[15];
ry(-0.821949467598637) q[16];
cx q[15],q[16];
ry(2.4536799486983742) q[15];
ry(-0.0007378814391190147) q[16];
cx q[15],q[16];
ry(-3.101224445307241) q[16];
ry(2.015176698096729) q[17];
cx q[16],q[17];
ry(2.076983848249946) q[16];
ry(-3.1394189684006935) q[17];
cx q[16],q[17];
ry(3.0191901645053303) q[17];
ry(-2.8047562481383803) q[18];
cx q[17],q[18];
ry(-3.135869654159083) q[17];
ry(-3.125066241260955) q[18];
cx q[17],q[18];
ry(0.5506857097640605) q[18];
ry(0.7029499050893051) q[19];
cx q[18],q[19];
ry(-0.28057158655738323) q[18];
ry(-1.617354421570612) q[19];
cx q[18],q[19];
ry(-2.489898884319983) q[0];
ry(-1.0664200941775475) q[1];
ry(1.5715330308084905) q[2];
ry(1.5681783383438426) q[3];
ry(-1.5716622923573111) q[4];
ry(1.5697789190889635) q[5];
ry(-1.5806316624325358) q[6];
ry(0.23576431197373712) q[7];
ry(0.29743171149786074) q[8];
ry(-1.5645177333123208) q[9];
ry(1.5844495814033779) q[10];
ry(-1.5720913185368872) q[11];
ry(0.4535937676033562) q[12];
ry(-1.5709690810292316) q[13];
ry(-1.570685486175969) q[14];
ry(2.8317680151370705) q[15];
ry(0.0390864479986598) q[16];
ry(3.0182123300930166) q[17];
ry(-3.0998653437485633) q[18];
ry(-2.6078072056772283) q[19];