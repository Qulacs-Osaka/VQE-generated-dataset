OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.14588901410260746) q[0];
ry(2.7256863566761513) q[1];
cx q[0],q[1];
ry(-0.41758039315155354) q[0];
ry(-0.26857312352301665) q[1];
cx q[0],q[1];
ry(-1.868624945873945) q[1];
ry(-1.6443685740345) q[2];
cx q[1],q[2];
ry(-0.752409491441575) q[1];
ry(-3.077648974920621) q[2];
cx q[1],q[2];
ry(-0.9604650331247315) q[2];
ry(1.7168900659847992) q[3];
cx q[2],q[3];
ry(1.7462246551497533) q[2];
ry(0.045988775147292396) q[3];
cx q[2],q[3];
ry(-1.4903886696142938) q[3];
ry(2.0923430890608983) q[4];
cx q[3],q[4];
ry(3.1101337547160077) q[3];
ry(0.4566664617119178) q[4];
cx q[3],q[4];
ry(3.081753401739897) q[4];
ry(1.5517850866242189) q[5];
cx q[4],q[5];
ry(3.0269154394570936) q[4];
ry(0.004524614089293884) q[5];
cx q[4],q[5];
ry(-0.9396345677235122) q[5];
ry(-1.8745608352417102) q[6];
cx q[5],q[6];
ry(0.2973308490553462) q[5];
ry(2.8759754900773364) q[6];
cx q[5],q[6];
ry(-2.341654334232058) q[6];
ry(1.6673256080488539) q[7];
cx q[6],q[7];
ry(-0.3620031092416456) q[6];
ry(3.1067936376933973) q[7];
cx q[6],q[7];
ry(-1.553310026496744) q[7];
ry(-1.9534646417244312) q[8];
cx q[7],q[8];
ry(3.038913394957445) q[7];
ry(-0.1756518002970321) q[8];
cx q[7],q[8];
ry(1.2467853555891566) q[8];
ry(0.1634163056137865) q[9];
cx q[8],q[9];
ry(-0.3758254340931826) q[8];
ry(-2.0376441546147315) q[9];
cx q[8],q[9];
ry(1.2083580515652692) q[9];
ry(2.8199987740804833) q[10];
cx q[9],q[10];
ry(1.8695132186815895) q[9];
ry(-1.069920894390883) q[10];
cx q[9],q[10];
ry(1.4890267387262748) q[10];
ry(1.901551067770061) q[11];
cx q[10],q[11];
ry(0.7599787920392373) q[10];
ry(-2.2470618824324213) q[11];
cx q[10],q[11];
ry(2.8930835586266834) q[11];
ry(-1.5723670029387486) q[12];
cx q[11],q[12];
ry(2.924107874775884) q[11];
ry(0.0001117563045562875) q[12];
cx q[11],q[12];
ry(1.0282374603132591) q[12];
ry(3.067885294995249) q[13];
cx q[12],q[13];
ry(-0.17411722713701927) q[12];
ry(-0.32139434634538316) q[13];
cx q[12],q[13];
ry(-1.8762697789599336) q[13];
ry(1.4030341869926568) q[14];
cx q[13],q[14];
ry(-0.9562372264565653) q[13];
ry(1.3487138560879712) q[14];
cx q[13],q[14];
ry(0.07313634956368976) q[14];
ry(-1.8594106335051084) q[15];
cx q[14],q[15];
ry(2.9067093587909807) q[14];
ry(1.2354676048311317) q[15];
cx q[14],q[15];
ry(0.16017989851786305) q[15];
ry(-0.2997938534347391) q[16];
cx q[15],q[16];
ry(1.9100269915879917) q[15];
ry(-2.201239111770695) q[16];
cx q[15],q[16];
ry(0.8191532311361279) q[16];
ry(1.569582765013621) q[17];
cx q[16],q[17];
ry(-1.5320433962102245) q[16];
ry(-0.00012336631965737652) q[17];
cx q[16],q[17];
ry(-3.087470497496737) q[17];
ry(-0.9989598458683145) q[18];
cx q[17],q[18];
ry(-0.40202964439660904) q[17];
ry(-3.0387967804542444) q[18];
cx q[17],q[18];
ry(-1.4179998994062255) q[18];
ry(-0.6175072131761876) q[19];
cx q[18],q[19];
ry(-3.0518223925545866) q[18];
ry(0.8357008197463817) q[19];
cx q[18],q[19];
ry(2.1142564466462623) q[0];
ry(2.6899220121835454) q[1];
cx q[0],q[1];
ry(1.4168743093047222) q[0];
ry(1.2334654215759828) q[1];
cx q[0],q[1];
ry(3.055673651800063) q[1];
ry(-0.7435917902539273) q[2];
cx q[1],q[2];
ry(-3.1368651941577412) q[1];
ry(-0.7693985095601998) q[2];
cx q[1],q[2];
ry(1.995415205773417) q[2];
ry(-2.1216488620271274) q[3];
cx q[2],q[3];
ry(0.4356552796325728) q[2];
ry(-0.03069573947501915) q[3];
cx q[2],q[3];
ry(-1.40015064442345) q[3];
ry(-1.408255872524388) q[4];
cx q[3],q[4];
ry(0.15492733587097085) q[3];
ry(-2.3058100566364548) q[4];
cx q[3],q[4];
ry(-2.559398016799777) q[4];
ry(1.9436495817063093) q[5];
cx q[4],q[5];
ry(2.946997579674416) q[4];
ry(-0.002358629999301698) q[5];
cx q[4],q[5];
ry(1.5658827540081424) q[5];
ry(-2.7737858035389813) q[6];
cx q[5],q[6];
ry(0.933695831635997) q[5];
ry(-2.6856867255166192) q[6];
cx q[5],q[6];
ry(-1.950753882360491) q[6];
ry(2.701090398614643) q[7];
cx q[6],q[7];
ry(-0.957907952538078) q[6];
ry(-2.915519037557084) q[7];
cx q[6],q[7];
ry(1.5172287114030025) q[7];
ry(2.1966392367130005) q[8];
cx q[7],q[8];
ry(-3.140821819098131) q[7];
ry(0.00015286232886158047) q[8];
cx q[7],q[8];
ry(-1.4324625664175004) q[8];
ry(2.5849423264320466) q[9];
cx q[8],q[9];
ry(0.12188386216824387) q[8];
ry(-2.747772994860022) q[9];
cx q[8],q[9];
ry(-1.5132112302759362) q[9];
ry(-2.236000394696176) q[10];
cx q[9],q[10];
ry(-2.0150726260086618) q[9];
ry(0.5290273711803275) q[10];
cx q[9],q[10];
ry(-1.0337003135034197) q[10];
ry(-2.0276525000294745) q[11];
cx q[10],q[11];
ry(3.1018967397349537) q[10];
ry(2.1291099409551717) q[11];
cx q[10],q[11];
ry(2.793566008794709) q[11];
ry(-2.8594953917182195) q[12];
cx q[11],q[12];
ry(0.00035097249155757737) q[11];
ry(3.14155431440944) q[12];
cx q[11],q[12];
ry(3.055815706210636) q[12];
ry(-0.7018885150690588) q[13];
cx q[12],q[13];
ry(0.8331280208826513) q[12];
ry(2.4751340846789986) q[13];
cx q[12],q[13];
ry(0.054769427418917785) q[13];
ry(-1.5020061056304748) q[14];
cx q[13],q[14];
ry(-2.093923077014056) q[13];
ry(-0.0021497341022014287) q[14];
cx q[13],q[14];
ry(-2.362433098154623) q[14];
ry(-2.539972309307397) q[15];
cx q[14],q[15];
ry(-0.9749667067040025) q[14];
ry(-2.929714922752105) q[15];
cx q[14],q[15];
ry(1.578579323074702) q[15];
ry(-1.628130378607546) q[16];
cx q[15],q[16];
ry(2.8980427239386035) q[15];
ry(-1.731301838358787) q[16];
cx q[15],q[16];
ry(-2.202855599457677) q[16];
ry(-1.4991321733222085) q[17];
cx q[16],q[17];
ry(-3.1408131938564248) q[16];
ry(2.124049896932447) q[17];
cx q[16],q[17];
ry(0.05710363854429623) q[17];
ry(1.7477970576839994) q[18];
cx q[17],q[18];
ry(-0.10422085428479023) q[17];
ry(-0.011807952985002144) q[18];
cx q[17],q[18];
ry(0.6477152108577203) q[18];
ry(3.074393411916845) q[19];
cx q[18],q[19];
ry(2.1993332967290042) q[18];
ry(1.1736724255667368) q[19];
cx q[18],q[19];
ry(3.1028271217488905) q[0];
ry(0.5144733794090159) q[1];
cx q[0],q[1];
ry(2.484364250509541) q[0];
ry(0.22882232003192726) q[1];
cx q[0],q[1];
ry(-2.1832131257431096) q[1];
ry(-1.532416418652393) q[2];
cx q[1],q[2];
ry(0.015060955979536259) q[1];
ry(0.0586559398161793) q[2];
cx q[1],q[2];
ry(0.05334599365698553) q[2];
ry(-1.5708195174676511) q[3];
cx q[2],q[3];
ry(-0.47474841761544945) q[2];
ry(0.09719363096701095) q[3];
cx q[2],q[3];
ry(-1.5168459449971854) q[3];
ry(-1.9253012167950745) q[4];
cx q[3],q[4];
ry(-0.012471089669710243) q[3];
ry(1.1274338319095172) q[4];
cx q[3],q[4];
ry(-1.8173716615133892) q[4];
ry(2.4127080720087455) q[5];
cx q[4],q[5];
ry(-0.6425245189188109) q[4];
ry(0.5811765290056208) q[5];
cx q[4],q[5];
ry(-1.85495062694077) q[5];
ry(3.0194953961910187) q[6];
cx q[5],q[6];
ry(2.301120494871126) q[5];
ry(-2.709562290740479) q[6];
cx q[5],q[6];
ry(-0.9265170590486633) q[6];
ry(0.9516363277397462) q[7];
cx q[6],q[7];
ry(2.991242535370943) q[6];
ry(-3.0195121592125154) q[7];
cx q[6],q[7];
ry(-2.2075876355812154) q[7];
ry(0.8644026566694184) q[8];
cx q[7],q[8];
ry(-3.139780232183341) q[7];
ry(-3.1369890415586155) q[8];
cx q[7],q[8];
ry(1.0918922380941796) q[8];
ry(-1.836163293192552) q[9];
cx q[8],q[9];
ry(-0.5991757334578915) q[8];
ry(-0.47770543865299725) q[9];
cx q[8],q[9];
ry(1.3455267762070338) q[9];
ry(0.9779329882389507) q[10];
cx q[9],q[10];
ry(2.4484446748707627) q[9];
ry(2.24026895829505) q[10];
cx q[9],q[10];
ry(1.1122574039708406) q[10];
ry(1.4570252722259724) q[11];
cx q[10],q[11];
ry(2.2722412426769782) q[10];
ry(-1.319989032941082) q[11];
cx q[10],q[11];
ry(-0.344327161527784) q[11];
ry(-1.3128497967186252) q[12];
cx q[11],q[12];
ry(-3.110657447186759) q[11];
ry(1.0751872440310393) q[12];
cx q[11],q[12];
ry(-2.752873484639994) q[12];
ry(1.0145718336239027) q[13];
cx q[12],q[13];
ry(-1.6215023026246729) q[12];
ry(-1.5774510302435376e-06) q[13];
cx q[12],q[13];
ry(2.039353095484202) q[13];
ry(2.7242574105952317) q[14];
cx q[13],q[14];
ry(-0.010819158723443092) q[13];
ry(3.1403407061183195) q[14];
cx q[13],q[14];
ry(-1.5264887198430657) q[14];
ry(-2.2194996036165957) q[15];
cx q[14],q[15];
ry(-2.9190629852027326) q[14];
ry(1.7822998283001281) q[15];
cx q[14],q[15];
ry(1.5804691603012444) q[15];
ry(1.5677486938637593) q[16];
cx q[15],q[16];
ry(1.471833752124068) q[15];
ry(0.005491662435237675) q[16];
cx q[15],q[16];
ry(-2.6968703779692094) q[16];
ry(-2.776091073616687) q[17];
cx q[16],q[17];
ry(-2.944858738076817) q[16];
ry(3.1210890556841946) q[17];
cx q[16],q[17];
ry(1.2806760552491205) q[17];
ry(-0.9058676339021794) q[18];
cx q[17],q[18];
ry(3.103150453015524) q[17];
ry(0.10105041981034121) q[18];
cx q[17],q[18];
ry(-2.6926260731410125) q[18];
ry(1.705254761678069) q[19];
cx q[18],q[19];
ry(2.2885963323984533) q[18];
ry(1.465858031506933) q[19];
cx q[18],q[19];
ry(-2.5016702561764856) q[0];
ry(0.5280651005496165) q[1];
cx q[0],q[1];
ry(0.13508715448825534) q[0];
ry(-0.824140774380089) q[1];
cx q[0],q[1];
ry(-1.529919667002468) q[1];
ry(-0.1936119479538026) q[2];
cx q[1],q[2];
ry(-0.012340868022441498) q[1];
ry(-0.027740751391582646) q[2];
cx q[1],q[2];
ry(2.9800883397541744) q[2];
ry(-1.609815624680748) q[3];
cx q[2],q[3];
ry(-1.7005229205715955) q[2];
ry(-1.1757706766583595) q[3];
cx q[2],q[3];
ry(-1.7648915948913793) q[3];
ry(-1.4498151705549676) q[4];
cx q[3],q[4];
ry(2.6344959001659207) q[3];
ry(-2.582556374399002) q[4];
cx q[3],q[4];
ry(-1.1148123557314047) q[4];
ry(1.3101981219770744) q[5];
cx q[4],q[5];
ry(0.003075725535911844) q[4];
ry(3.1161391854724063) q[5];
cx q[4],q[5];
ry(0.7681519657509347) q[5];
ry(2.2608623351607213) q[6];
cx q[5],q[6];
ry(2.270771058054934) q[5];
ry(0.03618697386295988) q[6];
cx q[5],q[6];
ry(-1.4375943047802275) q[6];
ry(1.6083494610335771) q[7];
cx q[6],q[7];
ry(-0.7146215681823709) q[6];
ry(2.7958296470723036) q[7];
cx q[6],q[7];
ry(-1.7195124221127147) q[7];
ry(-1.3343431251923406) q[8];
cx q[7],q[8];
ry(-0.08037137679394406) q[7];
ry(0.45272960880158836) q[8];
cx q[7],q[8];
ry(-1.991892812264079) q[8];
ry(1.1306403276240315) q[9];
cx q[8],q[9];
ry(-2.819781013374919) q[8];
ry(3.1262233645072777) q[9];
cx q[8],q[9];
ry(-0.9935725222171934) q[9];
ry(-1.038333934576274) q[10];
cx q[9],q[10];
ry(1.6287538506797778) q[9];
ry(0.5909680033427467) q[10];
cx q[9],q[10];
ry(-2.3844151177756197) q[10];
ry(-1.2150009111542215) q[11];
cx q[10],q[11];
ry(3.1382620630382076) q[10];
ry(3.140638498662925) q[11];
cx q[10],q[11];
ry(-1.1960688478588) q[11];
ry(0.7415824263487929) q[12];
cx q[11],q[12];
ry(3.1321328888108835) q[11];
ry(-1.0819014434217316) q[12];
cx q[11],q[12];
ry(2.4829727451006427) q[12];
ry(2.5224597102168747) q[13];
cx q[12],q[13];
ry(-2.6979029195440387) q[12];
ry(-0.3352095227884737) q[13];
cx q[12],q[13];
ry(-1.929985347664756) q[13];
ry(2.695550663263044) q[14];
cx q[13],q[14];
ry(2.386538130076742) q[13];
ry(-2.086020952916816) q[14];
cx q[13],q[14];
ry(-0.08871996912658911) q[14];
ry(2.1036208018537743) q[15];
cx q[14],q[15];
ry(-1.90813935562191) q[14];
ry(-3.1238500655618124) q[15];
cx q[14],q[15];
ry(-1.8072062734543408) q[15];
ry(1.3531167553030958) q[16];
cx q[15],q[16];
ry(-0.0004923071989836329) q[15];
ry(-0.00048227565481973045) q[16];
cx q[15],q[16];
ry(-2.6467531014868113) q[16];
ry(0.7177860420158483) q[17];
cx q[16],q[17];
ry(0.15227213555357696) q[16];
ry(-1.6659316596142089) q[17];
cx q[16],q[17];
ry(2.093806989888344) q[17];
ry(0.9261887221682388) q[18];
cx q[17],q[18];
ry(2.992799128199315) q[17];
ry(-3.010292073474826) q[18];
cx q[17],q[18];
ry(-1.5773946523415536) q[18];
ry(-0.7316681527227633) q[19];
cx q[18],q[19];
ry(0.664560579698282) q[18];
ry(-0.9473524987516617) q[19];
cx q[18],q[19];
ry(2.2784886894328826) q[0];
ry(0.8724445509739258) q[1];
cx q[0],q[1];
ry(-3.0900841184337526) q[0];
ry(0.7775593583225975) q[1];
cx q[0],q[1];
ry(0.5368598713187035) q[1];
ry(2.7739490454570728) q[2];
cx q[1],q[2];
ry(-0.3398524167010392) q[1];
ry(1.2617331953236004) q[2];
cx q[1],q[2];
ry(-1.5634055747762554) q[2];
ry(1.7420971041781872) q[3];
cx q[2],q[3];
ry(3.082539113214969) q[2];
ry(-3.1347447987918193) q[3];
cx q[2],q[3];
ry(2.1581512927404987) q[3];
ry(2.9210084029703776) q[4];
cx q[3],q[4];
ry(-2.974684962682004) q[3];
ry(-3.1187833941820644) q[4];
cx q[3],q[4];
ry(-1.6956895402458998) q[4];
ry(0.5363381881750948) q[5];
cx q[4],q[5];
ry(-0.0011229630417677328) q[4];
ry(0.007547172069469359) q[5];
cx q[4],q[5];
ry(2.0860369576628894) q[5];
ry(0.9419443102772833) q[6];
cx q[5],q[6];
ry(3.082116305069505) q[5];
ry(-0.018534982306490196) q[6];
cx q[5],q[6];
ry(0.021490921939381735) q[6];
ry(-2.27583615242039) q[7];
cx q[6],q[7];
ry(-3.1301686224192427) q[6];
ry(-3.141054789971111) q[7];
cx q[6],q[7];
ry(-2.0492375337135558) q[7];
ry(1.1969760983985853) q[8];
cx q[7],q[8];
ry(3.022740309812714) q[7];
ry(-3.114349611209745) q[8];
cx q[7],q[8];
ry(1.2857201190769119) q[8];
ry(2.289238950752442) q[9];
cx q[8],q[9];
ry(-2.676274243915101) q[8];
ry(-0.43250028204141877) q[9];
cx q[8],q[9];
ry(1.7492460658882811) q[9];
ry(0.8040230548534624) q[10];
cx q[9],q[10];
ry(-0.6974430008288253) q[9];
ry(2.3822376253201774) q[10];
cx q[9],q[10];
ry(-0.13237889338466413) q[10];
ry(0.6735697071993094) q[11];
cx q[10],q[11];
ry(-0.00704228368298132) q[10];
ry(-2.9122390509106544) q[11];
cx q[10],q[11];
ry(-2.9417618528798366) q[11];
ry(1.576180747500116) q[12];
cx q[11],q[12];
ry(0.6631684951951139) q[11];
ry(-0.5295197411910673) q[12];
cx q[11],q[12];
ry(-0.6684915185869391) q[12];
ry(1.450983833468701) q[13];
cx q[12],q[13];
ry(2.215772120379839) q[12];
ry(2.8489104310248488) q[13];
cx q[12],q[13];
ry(1.8904132791834052) q[13];
ry(-0.29862430017646674) q[14];
cx q[13],q[14];
ry(3.140465023968393) q[13];
ry(-2.810208414939918) q[14];
cx q[13],q[14];
ry(-1.389492233167877) q[14];
ry(-0.030055012588411093) q[15];
cx q[14],q[15];
ry(2.7571544209253727) q[14];
ry(1.1398583142384942) q[15];
cx q[14],q[15];
ry(-0.40434188062680043) q[15];
ry(-0.5895864980920106) q[16];
cx q[15],q[16];
ry(0.07499358001110012) q[15];
ry(-0.44625548462113684) q[16];
cx q[15],q[16];
ry(-1.4935077676487787) q[16];
ry(-0.7339597402425848) q[17];
cx q[16],q[17];
ry(0.7240960890665087) q[16];
ry(-1.3800987663945512) q[17];
cx q[16],q[17];
ry(0.1206116428407357) q[17];
ry(2.9849658297334227) q[18];
cx q[17],q[18];
ry(0.5608509897544101) q[17];
ry(-0.18817353369776324) q[18];
cx q[17],q[18];
ry(-2.180010290312822) q[18];
ry(-0.6426028092981404) q[19];
cx q[18],q[19];
ry(3.106114441827784) q[18];
ry(-0.09648527368274708) q[19];
cx q[18],q[19];
ry(-1.5575031071677543) q[0];
ry(-1.5836213608482241) q[1];
cx q[0],q[1];
ry(-2.597361737266859) q[0];
ry(-2.969794475331066) q[1];
cx q[0],q[1];
ry(1.5820727886248196) q[1];
ry(-1.2990751424146734) q[2];
cx q[1],q[2];
ry(-3.013677647443287) q[1];
ry(-1.0672946533536898) q[2];
cx q[1],q[2];
ry(2.342254216206064) q[2];
ry(-0.4886147233116338) q[3];
cx q[2],q[3];
ry(0.02766552243613313) q[2];
ry(3.123968964254193) q[3];
cx q[2],q[3];
ry(0.3631001383346226) q[3];
ry(-1.0240184182157233) q[4];
cx q[3],q[4];
ry(-2.7834526618837048) q[3];
ry(0.2163079523207091) q[4];
cx q[3],q[4];
ry(-2.3462911033992557) q[4];
ry(-0.009298310669748271) q[5];
cx q[4],q[5];
ry(0.20273113780145202) q[4];
ry(3.1151572611664218) q[5];
cx q[4],q[5];
ry(-2.4034634400828163) q[5];
ry(2.336074025600472) q[6];
cx q[5],q[6];
ry(-3.1181920067176976) q[5];
ry(0.0024307280236092544) q[6];
cx q[5],q[6];
ry(-2.695766930226888) q[6];
ry(1.665225001354363) q[7];
cx q[6],q[7];
ry(-0.07846555150517244) q[6];
ry(3.1339242994955825) q[7];
cx q[6],q[7];
ry(-0.17525647316492599) q[7];
ry(2.4975189660337294) q[8];
cx q[7],q[8];
ry(-3.128114185060519) q[7];
ry(-3.136588957570809) q[8];
cx q[7],q[8];
ry(0.8613473716674473) q[8];
ry(-0.9028587857074654) q[9];
cx q[8],q[9];
ry(0.0609465078626803) q[8];
ry(0.020797863198291466) q[9];
cx q[8],q[9];
ry(1.3529996325407403) q[9];
ry(2.7862574648788687) q[10];
cx q[9],q[10];
ry(2.826585791898932) q[9];
ry(2.7995477534842426) q[10];
cx q[9],q[10];
ry(2.879409834119714) q[10];
ry(-1.086550344197042) q[11];
cx q[10],q[11];
ry(-3.134097193760972) q[10];
ry(0.08739336467526448) q[11];
cx q[10],q[11];
ry(2.222864495980182) q[11];
ry(2.3980736850210134) q[12];
cx q[11],q[12];
ry(-0.020243298432129464) q[11];
ry(0.021084345856356424) q[12];
cx q[11],q[12];
ry(1.9231513840466832) q[12];
ry(-2.4049740636920847) q[13];
cx q[12],q[13];
ry(-0.818207289351033) q[12];
ry(-0.2531091403761376) q[13];
cx q[12],q[13];
ry(0.10068702700466621) q[13];
ry(-0.1669774035415097) q[14];
cx q[13],q[14];
ry(2.1561940859172433) q[13];
ry(2.7837318736795282) q[14];
cx q[13],q[14];
ry(-1.9569643287002858) q[14];
ry(-1.5753870775737482) q[15];
cx q[14],q[15];
ry(-0.23245750528047893) q[14];
ry(3.138447483790404) q[15];
cx q[14],q[15];
ry(-1.5732571930055106) q[15];
ry(-1.572676826406572) q[16];
cx q[15],q[16];
ry(-0.12655022743937216) q[15];
ry(0.0698342072298388) q[16];
cx q[15],q[16];
ry(-1.5597018316283013) q[16];
ry(-1.6560452152688567) q[17];
cx q[16],q[17];
ry(2.887075429794707) q[16];
ry(-1.9389795403138708) q[17];
cx q[16],q[17];
ry(-2.9407321807609437) q[17];
ry(-1.523823029018497) q[18];
cx q[17],q[18];
ry(0.351042556336691) q[17];
ry(0.025132243663315474) q[18];
cx q[17],q[18];
ry(2.424682413824996) q[18];
ry(2.6656651391274986) q[19];
cx q[18],q[19];
ry(0.4862289327239724) q[18];
ry(-2.685330902791463) q[19];
cx q[18],q[19];
ry(-1.3028989423840054) q[0];
ry(3.1198899457805824) q[1];
ry(0.6047147516458384) q[2];
ry(0.404802053696077) q[3];
ry(2.9669663427347186) q[4];
ry(0.8378300662620211) q[5];
ry(0.24275475628844223) q[6];
ry(-1.9020545598155851) q[7];
ry(1.6084234590738578) q[8];
ry(-2.443546282189417) q[9];
ry(3.1258946063146644) q[10];
ry(2.9910567116167246) q[11];
ry(2.5130846768423107) q[12];
ry(-2.932832968630621) q[13];
ry(1.2485839459774706) q[14];
ry(0.0028543949881812926) q[15];
ry(0.03232899141899992) q[16];
ry(-1.802973412311105) q[17];
ry(2.951032530253793) q[18];
ry(-2.8908471894100507) q[19];