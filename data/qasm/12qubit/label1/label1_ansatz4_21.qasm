OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.58344348374547) q[0];
rz(2.9129682396111534) q[0];
ry(-1.0904304063106303) q[1];
rz(-1.8135871578965599) q[1];
ry(0.6757542448403139) q[2];
rz(-1.7399336599648203) q[2];
ry(2.783045947618361) q[3];
rz(-2.4694316932045095) q[3];
ry(-0.16199889279881763) q[4];
rz(2.267437104359601) q[4];
ry(0.6338537405804263) q[5];
rz(1.4454929919350397) q[5];
ry(-0.07027442333431744) q[6];
rz(1.244467656588377) q[6];
ry(-3.081645822026711) q[7];
rz(1.165312177612475) q[7];
ry(-1.5792817674765331) q[8];
rz(2.614514046387947) q[8];
ry(1.5843486018137989) q[9];
rz(-0.8268309328543308) q[9];
ry(-0.0039400267164351735) q[10];
rz(1.251298675750112) q[10];
ry(-1.560194915989916) q[11];
rz(1.5766488141256265) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.7913203468309158) q[0];
rz(0.9649211625160561) q[0];
ry(1.3818079009794764) q[1];
rz(2.051052704086674) q[1];
ry(-1.037829575402493) q[2];
rz(0.7435336885076922) q[2];
ry(-0.9115660030277271) q[3];
rz(-0.7004024416239281) q[3];
ry(-1.736987180193366) q[4];
rz(0.39211201258907025) q[4];
ry(0.9754649774424875) q[5];
rz(1.9350265202351935) q[5];
ry(-3.13707159328455) q[6];
rz(-1.2793361364494777) q[6];
ry(3.133513431319878) q[7];
rz(0.5183438579068577) q[7];
ry(0.0301507108073071) q[8];
rz(-1.325866495372404) q[8];
ry(-0.06190535458928803) q[9];
rz(0.20956229702898008) q[9];
ry(3.134893342302693) q[10];
rz(0.6763317830250524) q[10];
ry(1.5744703399602207) q[11];
rz(-0.9128141573934858) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.0659308126485507) q[0];
rz(-1.2144627229099132) q[0];
ry(-1.0161371656644123) q[1];
rz(1.210025934171017) q[1];
ry(-3.0943364836151424) q[2];
rz(1.235104166359706) q[2];
ry(0.19270191745415113) q[3];
rz(0.8819865926748846) q[3];
ry(-0.2576426306824012) q[4];
rz(-0.6562961870759398) q[4];
ry(-2.760395866961491) q[5];
rz(-2.801041500763494) q[5];
ry(1.5486167776614082) q[6];
rz(-0.01566198804063923) q[6];
ry(-1.5861683627571868) q[7];
rz(0.0013368588906883608) q[7];
ry(-1.770393976353052) q[8];
rz(-1.5694552851142527) q[8];
ry(0.3454780539403437) q[9];
rz(0.3933510987215611) q[9];
ry(0.5132742656888548) q[10];
rz(-1.2068131601667367) q[10];
ry(2.4787540481455745) q[11];
rz(3.121878575218838) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.0406421156678478) q[0];
rz(-0.6974376259760646) q[0];
ry(1.8628991474314038) q[1];
rz(0.25052414935687256) q[1];
ry(-2.3159379690436155) q[2];
rz(-2.9810956270319653) q[2];
ry(0.024758485091294528) q[3];
rz(-2.6825070376277074) q[3];
ry(2.6213832000053925) q[4];
rz(2.7088957422285134) q[4];
ry(1.748368206739653) q[5];
rz(2.1800631820995755) q[5];
ry(-1.1024092412663378) q[6];
rz(-2.7523896765649565) q[6];
ry(1.7827625182805198) q[7];
rz(2.7393745098462983) q[7];
ry(-3.136907574757844) q[8];
rz(1.4339940911087679) q[8];
ry(-3.1289701514818) q[9];
rz(-1.5465513623005718) q[9];
ry(0.4145780423455643) q[10];
rz(-0.5040735970515815) q[10];
ry(-0.5450071941468733) q[11];
rz(-2.071936680844612) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.325364634120897) q[0];
rz(-0.2844668803933867) q[0];
ry(0.9357497057050983) q[1];
rz(-1.517353652925487) q[1];
ry(-2.2038047466192436) q[2];
rz(2.5747195876238926) q[2];
ry(-0.9035717334494671) q[3];
rz(-0.7444491700875524) q[3];
ry(-1.063056739049539) q[4];
rz(1.3750242837419009) q[4];
ry(2.8577746053384594) q[5];
rz(-0.7307038855786089) q[5];
ry(-0.03898537546345793) q[6];
rz(-0.7128025033731419) q[6];
ry(0.032208303444604354) q[7];
rz(0.7026226181408113) q[7];
ry(2.814616038808615) q[8];
rz(1.1056169790403385) q[8];
ry(-2.8305792643008845) q[9];
rz(0.7774730374100772) q[9];
ry(-0.5740677777590308) q[10];
rz(-1.920473695499214) q[10];
ry(-2.978466825769092) q[11];
rz(-0.07291356457762482) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.6697652790193009) q[0];
rz(-2.0434172778504784) q[0];
ry(-0.697464337579262) q[1];
rz(2.6490791646087803) q[1];
ry(2.7811182955595117) q[2];
rz(-0.43679185116177427) q[2];
ry(1.062029577969037) q[3];
rz(2.7380110539386076) q[3];
ry(-1.9313983362049028) q[4];
rz(-0.14423796251979581) q[4];
ry(1.3604285125626154) q[5];
rz(-2.7978782850780406) q[5];
ry(0.039287605506375854) q[6];
rz(-1.401538509360434) q[6];
ry(-3.1158807394722485) q[7];
rz(1.726629115969267) q[7];
ry(0.007375458604550466) q[8];
rz(1.4627325788455807) q[8];
ry(0.004012155980031523) q[9];
rz(3.088296696443516) q[9];
ry(-0.5721579066978819) q[10];
rz(1.0971514583209083) q[10];
ry(-1.6884718919378745) q[11];
rz(1.4976560354043125) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.0310794053421355) q[0];
rz(-0.617707384615823) q[0];
ry(-0.38888108890782647) q[1];
rz(-1.1299928388031208) q[1];
ry(-3.0371500301472585) q[2];
rz(-1.1741139448472344) q[2];
ry(0.38437359822875555) q[3];
rz(0.13450178383322786) q[3];
ry(-2.7065199714652355) q[4];
rz(0.32844743601460474) q[4];
ry(2.5863116836543782) q[5];
rz(1.9466255598536855) q[5];
ry(0.4795480428748673) q[6];
rz(-1.2000801837519968) q[6];
ry(-0.5260240777792626) q[7];
rz(-0.327911461368338) q[7];
ry(-1.2755591214582132) q[8];
rz(-0.032705482090666406) q[8];
ry(1.1560196311525757) q[9];
rz(-1.5297485760294707) q[9];
ry(-1.647714150553834) q[10];
rz(-0.08973382752259323) q[10];
ry(-0.6669407665597198) q[11];
rz(-1.381060092215411) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.372366228834512) q[0];
rz(-0.49704095949721605) q[0];
ry(2.918417689335693) q[1];
rz(-0.8614226618695017) q[1];
ry(2.7708983593414214) q[2];
rz(-2.171482524377198) q[2];
ry(1.3243411369698546) q[3];
rz(-1.0108544167189766) q[3];
ry(0.8648346827408356) q[4];
rz(0.24854498375780876) q[4];
ry(0.09390113265031903) q[5];
rz(0.406310959655709) q[5];
ry(-0.01938442267559874) q[6];
rz(2.860340600437684) q[6];
ry(-3.1361388408273516) q[7];
rz(1.5904919647845859) q[7];
ry(0.6697265213542644) q[8];
rz(-2.1378643620356357) q[8];
ry(2.625655263924487) q[9];
rz(-1.1169282888162115) q[9];
ry(-2.0454620800132) q[10];
rz(1.0919576705616) q[10];
ry(-0.9732629712087881) q[11];
rz(-3.0081117107197777) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.673469844615468) q[0];
rz(0.7206172110218644) q[0];
ry(-2.6417079966040165) q[1];
rz(-2.712427094875565) q[1];
ry(-1.400295820784993) q[2];
rz(-1.3755214589919582) q[2];
ry(2.7237129476189206) q[3];
rz(1.9637529278933128) q[3];
ry(2.4857122780992795) q[4];
rz(-2.3258961136235476) q[4];
ry(2.346708401421049) q[5];
rz(-1.0142998578192328) q[5];
ry(3.129065431072198) q[6];
rz(-1.608412663230747) q[6];
ry(0.008449750789258916) q[7];
rz(1.086051573288286) q[7];
ry(-2.2944930741604628) q[8];
rz(-2.7343031721005437) q[8];
ry(-1.7483941164399552) q[9];
rz(-0.24747907646267242) q[9];
ry(-1.5850466737403088) q[10];
rz(-1.5136011136688063) q[10];
ry(-1.9137116025452174) q[11];
rz(-0.141141059038877) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.78307936743887) q[0];
rz(-1.882928648494523) q[0];
ry(0.9968197411398715) q[1];
rz(-0.6608272544978117) q[1];
ry(0.1533146984559295) q[2];
rz(0.20950376810846258) q[2];
ry(1.1376981885358264) q[3];
rz(-1.6177853942961415) q[3];
ry(2.452691430071417) q[4];
rz(3.0986522300930015) q[4];
ry(-2.03194763865063) q[5];
rz(2.6057515267602724) q[5];
ry(-3.1400669061054596) q[6];
rz(1.9921282937883522) q[6];
ry(-3.140729523355907) q[7];
rz(-1.8679503624166163) q[7];
ry(1.3919200146011848) q[8];
rz(-1.6344083870120503) q[8];
ry(-1.6457503232061466) q[9];
rz(1.6327411716775624) q[9];
ry(2.0652582041122747) q[10];
rz(2.4335029732127413) q[10];
ry(-2.503419458025176) q[11];
rz(-0.010347178665033787) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.243747041710064) q[0];
rz(0.06658435596198264) q[0];
ry(0.3765775624248846) q[1];
rz(0.8783061510356642) q[1];
ry(-2.359637440458117) q[2];
rz(1.1619403139663362) q[2];
ry(0.4813234157057593) q[3];
rz(-2.708072507289073) q[3];
ry(-0.29163148985996656) q[4];
rz(0.4923329242846028) q[4];
ry(2.947603377817023) q[5];
rz(-0.5002798794995647) q[5];
ry(-0.004480829765470595) q[6];
rz(-2.1955392415841155) q[6];
ry(-3.136157566881875) q[7];
rz(2.7084965033331483) q[7];
ry(1.566890991215193) q[8];
rz(-1.9011410991523405) q[8];
ry(1.5246689924489356) q[9];
rz(1.871782334663377) q[9];
ry(1.950235137207413) q[10];
rz(1.8389314404913146) q[10];
ry(-1.8590117531366177) q[11];
rz(-2.4356458750207994) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.02533833030296595) q[0];
rz(-2.754736281901196) q[0];
ry(1.2003655281227659) q[1];
rz(-2.0515269053697107) q[1];
ry(-0.9774093317147186) q[2];
rz(-2.0796687294027993) q[2];
ry(-0.5318104438277756) q[3];
rz(0.6439589416767735) q[3];
ry(0.5410035200797427) q[4];
rz(1.3648573908405002) q[4];
ry(-2.348272178507267) q[5];
rz(-0.5132377289777388) q[5];
ry(-0.015391665145356724) q[6];
rz(0.4956768131903182) q[6];
ry(3.0508640622190724) q[7];
rz(-2.7791930346400546) q[7];
ry(0.9531337032295367) q[8];
rz(-0.6804677892413588) q[8];
ry(-2.141957985804594) q[9];
rz(-1.7886993717706527) q[9];
ry(-1.7145206968738118) q[10];
rz(0.12387027150177764) q[10];
ry(-1.701059198828643) q[11];
rz(0.10550173142308275) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.3661922621679006) q[0];
rz(-2.0759799325071833) q[0];
ry(1.1980886745741228) q[1];
rz(-2.6509499801311995) q[1];
ry(-0.2078409272897641) q[2];
rz(-1.4704983633945827) q[2];
ry(-1.8693636664100435) q[3];
rz(-2.0085032659755018) q[3];
ry(-0.7381397387735619) q[4];
rz(-0.992843779393552) q[4];
ry(-1.7858636725689372) q[5];
rz(-2.1505445469323696) q[5];
ry(3.135266646852838) q[6];
rz(0.5949365109796361) q[6];
ry(3.1281900826729756) q[7];
rz(-2.873453754787868) q[7];
ry(3.108182836384696) q[8];
rz(-0.7689222114363463) q[8];
ry(-0.07977536275527569) q[9];
rz(-0.08255436386382552) q[9];
ry(2.401446876552442) q[10];
rz(-0.24034386441472955) q[10];
ry(2.3224512347536135) q[11];
rz(1.0382689188281926) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.95960456052791) q[0];
rz(2.9437265537235087) q[0];
ry(-2.2476194713722046) q[1];
rz(-1.7763127013217246) q[1];
ry(-0.5663465169679361) q[2];
rz(-3.068155522993767) q[2];
ry(2.313975224365374) q[3];
rz(1.0305667534108855) q[3];
ry(-0.19816797855502966) q[4];
rz(0.8308191623958399) q[4];
ry(0.005107061933222899) q[5];
rz(-0.8553972946839944) q[5];
ry(-0.02028499929167938) q[6];
rz(1.4884324508522575) q[6];
ry(3.0383242483054422) q[7];
rz(-2.476711626066803) q[7];
ry(2.000360208219848) q[8];
rz(0.6689796969931256) q[8];
ry(1.0563150203081824) q[9];
rz(-3.066430069149677) q[9];
ry(-2.379107164884325) q[10];
rz(2.9928788032037836) q[10];
ry(1.86904045935655) q[11];
rz(-0.39289692755593997) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.37326315747223937) q[0];
rz(-1.3486123349245238) q[0];
ry(-1.8654023087752425) q[1];
rz(1.0072723245651451) q[1];
ry(0.17135843383488558) q[2];
rz(2.372750631740197) q[2];
ry(-1.201301276559966) q[3];
rz(-2.6090421715331376) q[3];
ry(0.3638542426920157) q[4];
rz(-1.1648079520233496) q[4];
ry(-0.34723930092025235) q[5];
rz(-0.07666283999416677) q[5];
ry(0.03144270490837453) q[6];
rz(2.1896569489715376) q[6];
ry(-3.1014850798340725) q[7];
rz(2.6892525041867272) q[7];
ry(3.0433288786125714) q[8];
rz(2.5646744529890197) q[8];
ry(3.139503413558041) q[9];
rz(-0.33615724165106364) q[9];
ry(2.1232026474551686) q[10];
rz(2.2357117629770773) q[10];
ry(0.6390638795949752) q[11];
rz(1.5652405891962093) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.1225035512189656) q[0];
rz(1.6008912292920456) q[0];
ry(0.5171387198097372) q[1];
rz(-0.8072857121845161) q[1];
ry(-0.15236705893589164) q[2];
rz(-2.116110480056201) q[2];
ry(-0.8221711391303392) q[3];
rz(-1.713981405242106) q[3];
ry(-3.0470272483936585) q[4];
rz(-0.6616366509395978) q[4];
ry(2.8822962788879014) q[5];
rz(1.0709097038595763) q[5];
ry(-3.0399201973627887) q[6];
rz(0.7726346665798861) q[6];
ry(0.0771621349872212) q[7];
rz(2.9209791993170913) q[7];
ry(2.8821844435263215) q[8];
rz(-2.1537604218883217) q[8];
ry(2.756563239872098) q[9];
rz(2.526999542301713) q[9];
ry(-2.5329972710014026) q[10];
rz(0.2121997645123992) q[10];
ry(1.118540864574972) q[11];
rz(0.8596185157103126) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.7410438850443655) q[0];
rz(1.3312622808456147) q[0];
ry(0.70128052104719) q[1];
rz(-1.1390100615330276) q[1];
ry(0.40965065989875044) q[2];
rz(2.5829208471097775) q[2];
ry(1.6148107832741903) q[3];
rz(0.3225415858300744) q[3];
ry(2.391834047338392) q[4];
rz(0.8212803647319717) q[4];
ry(2.583465308242149) q[5];
rz(-0.6068256152338566) q[5];
ry(-2.0097719306507145) q[6];
rz(-0.5037394654993816) q[6];
ry(-1.1085693627024271) q[7];
rz(-0.5124486413424458) q[7];
ry(-3.1070231790297034) q[8];
rz(-0.3784237498053736) q[8];
ry(-0.00989731900589854) q[9];
rz(2.0426244777587987) q[9];
ry(-0.8462551075992916) q[10];
rz(2.2717727407669197) q[10];
ry(2.6434413384021633) q[11];
rz(-2.1045075188119666) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.908336005369292) q[0];
rz(0.09474676825345796) q[0];
ry(1.5060369514202048) q[1];
rz(0.4214576896442628) q[1];
ry(-0.9934639857899896) q[2];
rz(-2.3320230106363398) q[2];
ry(-0.4385784507611924) q[3];
rz(2.9961813620632305) q[3];
ry(-0.3974027399895323) q[4];
rz(2.177685057099178) q[4];
ry(-3.065436131071368) q[5];
rz(-0.614119210793312) q[5];
ry(-1.4187549583622794) q[6];
rz(-3.1057346952089087) q[6];
ry(-1.5331011649009632) q[7];
rz(0.11795788852552125) q[7];
ry(2.9882718616521653) q[8];
rz(-1.6118436173929007) q[8];
ry(-0.04292101568090707) q[9];
rz(-0.12124086881869987) q[9];
ry(-0.03280422751218115) q[10];
rz(-0.9679426502660888) q[10];
ry(-0.9024517891794841) q[11];
rz(2.5558001217750705) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.9119280726994772) q[0];
rz(2.2940704946265162) q[0];
ry(0.2785847427940027) q[1];
rz(0.3787945218788149) q[1];
ry(-1.5075818779732493) q[2];
rz(-1.8052251620079154) q[2];
ry(-1.7932715485242372) q[3];
rz(-2.472469604062573) q[3];
ry(0.0027639831393990733) q[4];
rz(-1.5303668531310122) q[4];
ry(-0.0060679843759343655) q[5];
rz(2.7696228239616976) q[5];
ry(1.8715217292860222) q[6];
rz(-2.5258951607052738) q[6];
ry(-1.4225548028693933) q[7];
rz(-2.0944000948860313) q[7];
ry(1.5805883260418192) q[8];
rz(1.8411950918469344) q[8];
ry(-0.003859616222534125) q[9];
rz(-0.22761326707282256) q[9];
ry(-2.9587873683350225) q[10];
rz(2.951664332489792) q[10];
ry(1.6474361735780052) q[11];
rz(0.15306019291987158) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.5216984803113176) q[0];
rz(-0.6779321784313157) q[0];
ry(-2.6567350422167966) q[1];
rz(-0.9034829254208212) q[1];
ry(2.2243102898451754) q[2];
rz(2.7126519879262587) q[2];
ry(1.0397353502709294) q[3];
rz(-1.4884247889971274) q[3];
ry(-0.845956786786207) q[4];
rz(1.1957955944544292) q[4];
ry(2.5826892680419853) q[5];
rz(1.7060057648344857) q[5];
ry(-1.6867944200373015) q[6];
rz(-1.1288672325075384) q[6];
ry(1.9330488536505952) q[7];
rz(-2.257423736650339) q[7];
ry(-0.005891345763826549) q[8];
rz(-1.7129106940775243) q[8];
ry(-0.039098245324936776) q[9];
rz(-2.0912810388665464) q[9];
ry(1.5738017684351817) q[10];
rz(-3.134765051258576) q[10];
ry(-1.3034867250717417) q[11];
rz(1.213911881846771) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.39305132210147065) q[0];
rz(0.3789783914691032) q[0];
ry(2.0571358059569556) q[1];
rz(-2.4046216961076237) q[1];
ry(0.08112178320931039) q[2];
rz(-0.80076439227369) q[2];
ry(1.9973221250049962) q[3];
rz(0.15413323681752542) q[3];
ry(1.5640984448945237) q[4];
rz(-0.0021238798320224466) q[4];
ry(-1.5691705410557806) q[5];
rz(0.006319935660238585) q[5];
ry(-2.8277710797928632) q[6];
rz(2.417596511425241) q[6];
ry(-3.118589418714033) q[7];
rz(2.143578635655337) q[7];
ry(-1.5644121547329353) q[8];
rz(-0.977730987852595) q[8];
ry(0.035550039767260205) q[9];
rz(2.6275619897423512) q[9];
ry(-2.1517759951355275) q[10];
rz(-1.5505048040936162) q[10];
ry(0.26097756946686435) q[11];
rz(-2.4701212073436256) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.08488360728742333) q[0];
rz(-1.6343360095291302) q[0];
ry(-2.471057911164357) q[1];
rz(-2.9893096986849073) q[1];
ry(-2.382947850366607) q[2];
rz(-1.5083790726115256) q[2];
ry(1.0639967784206352) q[3];
rz(2.1301430068808687) q[3];
ry(1.5768145201829946) q[4];
rz(-0.4280957279409198) q[4];
ry(-1.5655301832370272) q[5];
rz(1.669403752687379) q[5];
ry(3.141293076226433) q[6];
rz(-2.693250816291456) q[6];
ry(-0.000987637053861512) q[7];
rz(2.391700291366231) q[7];
ry(-0.01664251642299587) q[8];
rz(-0.4931939437122823) q[8];
ry(0.0254778493923567) q[9];
rz(-2.3978488529032904) q[9];
ry(1.572538997605217) q[10];
rz(-1.1868149129097532) q[10];
ry(1.587663075999299) q[11];
rz(1.5592385741178503) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.9738974601323727) q[0];
rz(-1.2715533509165837) q[0];
ry(-2.1865216196032167) q[1];
rz(-0.07963363362786158) q[1];
ry(-1.182254954798401) q[2];
rz(-0.02484481175015763) q[2];
ry(1.9153291375303103) q[3];
rz(-0.24934848991055733) q[3];
ry(-0.37390589567003524) q[4];
rz(0.6293221494170369) q[4];
ry(-0.1181128664274796) q[5];
rz(2.5279834322726153) q[5];
ry(-1.4505414663233553) q[6];
rz(1.592564189847024) q[6];
ry(0.21069455022828357) q[7];
rz(2.1117221084876334) q[7];
ry(-3.094830167444354) q[8];
rz(1.5995677707087381) q[8];
ry(1.5610717101388831) q[9];
rz(-0.694319556025816) q[9];
ry(0.11101752249790506) q[10];
rz(-0.3638532306362667) q[10];
ry(1.0279554893052945) q[11];
rz(0.006291335863668479) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.9493403818008335) q[0];
rz(-0.7156351999469573) q[0];
ry(-2.1869970994811023) q[1];
rz(0.7768892773535483) q[1];
ry(-2.45898743846735) q[2];
rz(-0.133321504183324) q[2];
ry(-3.083601125874965) q[3];
rz(-1.4446682536728204) q[3];
ry(3.128356286713956) q[4];
rz(-3.0463345762643055) q[4];
ry(3.1385185726348324) q[5];
rz(1.24591887142466) q[5];
ry(-3.1415304550149528) q[6];
rz(3.1396698542627814) q[6];
ry(-0.0001579658936367171) q[7];
rz(1.8124456836878144) q[7];
ry(-3.0568756017741383) q[8];
rz(-0.0692957251413349) q[8];
ry(3.030488035212774) q[9];
rz(2.487389727506226) q[9];
ry(-1.5680266735390562) q[10];
rz(2.9519936395512043) q[10];
ry(-1.579305773317212) q[11];
rz(-2.6328709932306475) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.025359181630769313) q[0];
rz(-1.590439748108463) q[0];
ry(1.4575603633992016) q[1];
rz(-0.4164666320801471) q[1];
ry(-2.511036669860703) q[2];
rz(-1.7974775158607545) q[2];
ry(0.2910516183050889) q[3];
rz(2.164106668708092) q[3];
ry(-0.32109713665939055) q[4];
rz(2.235018529655407) q[4];
ry(1.7880849103614251) q[5];
rz(-1.925462234007803) q[5];
ry(-1.315069030150438) q[6];
rz(-2.3468680636349593) q[6];
ry(-1.703447467651947) q[7];
rz(2.32708833895006) q[7];
ry(1.5491236254311396) q[8];
rz(-2.5594987718459357) q[8];
ry(-1.5643125037081405) q[9];
rz(2.14161484719786) q[9];
ry(-0.132314910527219) q[10];
rz(0.7411604963944728) q[10];
ry(3.1408824526019483) q[11];
rz(2.5867439303645505) q[11];