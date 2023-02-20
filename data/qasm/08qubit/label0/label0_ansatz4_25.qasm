OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.09578023627156) q[0];
rz(-0.24485277382073445) q[0];
ry(-3.095618352614263) q[1];
rz(-0.4215684102819103) q[1];
ry(-1.7878135587778186) q[2];
rz(-0.4020280468286153) q[2];
ry(1.3604901947754513) q[3];
rz(-2.1362311582380684) q[3];
ry(-0.5207081091819313) q[4];
rz(-1.2268364064999124) q[4];
ry(-2.6657565606344806) q[5];
rz(2.6233643452595463) q[5];
ry(-1.1311350623728131) q[6];
rz(-2.873464716755953) q[6];
ry(2.011614018544629) q[7];
rz(-2.2120247745406836) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.718119034907379) q[0];
rz(-1.9074394009905284) q[0];
ry(1.1478272344355473) q[1];
rz(-2.1079815446082453) q[1];
ry(1.9493049366856035) q[2];
rz(2.461052104555941) q[2];
ry(0.7522138698415979) q[3];
rz(2.0426462032011106) q[3];
ry(2.058812059747006) q[4];
rz(0.4562929288018651) q[4];
ry(0.6302481450371342) q[5];
rz(0.21971380021256814) q[5];
ry(-2.787044880207315) q[6];
rz(1.739105376711839) q[6];
ry(-0.8520873424204833) q[7];
rz(-1.2812204496686914) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.7806760534891453) q[0];
rz(-0.548055958943097) q[0];
ry(-0.7031153350279077) q[1];
rz(-1.7804927045307712) q[1];
ry(1.656886066233354) q[2];
rz(-2.4243555700405723) q[2];
ry(2.4537603072888032) q[3];
rz(-1.97202994632998) q[3];
ry(-1.2336513148528274) q[4];
rz(0.7878469819910981) q[4];
ry(1.8824689068338083) q[5];
rz(-2.5622331944684316) q[5];
ry(2.066434777788544) q[6];
rz(-0.007330185557163028) q[6];
ry(-1.2699826230147822) q[7];
rz(1.5138859392990582) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.9441781015188447) q[0];
rz(0.8732959639869894) q[0];
ry(-2.8824589568541352) q[1];
rz(-1.054627784481081) q[1];
ry(0.4980263439341019) q[2];
rz(2.642704007586142) q[2];
ry(0.7527740940253763) q[3];
rz(-3.0366576963307828) q[3];
ry(-2.891616335207931) q[4];
rz(-3.109312580058584) q[4];
ry(0.04789333741659456) q[5];
rz(1.228801181743708) q[5];
ry(2.6308079636133233) q[6];
rz(-2.938729251218281) q[6];
ry(2.2024491891524622) q[7];
rz(-0.6281668743244326) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.4939687586876956) q[0];
rz(1.0580439915733755) q[0];
ry(0.4266671683098515) q[1];
rz(2.953599369905748) q[1];
ry(2.1012191006471492) q[2];
rz(1.4095227314521024) q[2];
ry(1.2230465351949884) q[3];
rz(0.10702005029768291) q[3];
ry(-0.4939914542240178) q[4];
rz(-1.4016373797424955) q[4];
ry(-2.411441005719072) q[5];
rz(-1.9517626405764195) q[5];
ry(0.8571124523677369) q[6];
rz(-1.5492387328342687) q[6];
ry(-1.8824163543880048) q[7];
rz(-2.9050909075908944) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.982066926211009) q[0];
rz(-0.8960880829549034) q[0];
ry(-1.891144624083121) q[1];
rz(-0.8925367221496233) q[1];
ry(0.2009112772723555) q[2];
rz(-0.8612064555126199) q[2];
ry(-2.728251156450378) q[3];
rz(-1.1350155151334522) q[3];
ry(-2.1392804889803236) q[4];
rz(-1.878939746109884) q[4];
ry(0.23844967581134568) q[5];
rz(1.8776358068816064) q[5];
ry(1.5261010428353987) q[6];
rz(-1.9307254465703039) q[6];
ry(-1.4677644656850184) q[7];
rz(1.2141527847492) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5330099632666485) q[0];
rz(3.120262634192527) q[0];
ry(0.5963668906093049) q[1];
rz(-2.239511972873019) q[1];
ry(2.997523127842052) q[2];
rz(1.8377307777169374) q[2];
ry(-2.2810982846765873) q[3];
rz(1.2571966403001014) q[3];
ry(-0.8332346899918734) q[4];
rz(2.5362103435478662) q[4];
ry(-1.7303476337039103) q[5];
rz(-1.8837407831397606) q[5];
ry(2.6117526090350225) q[6];
rz(-0.7800411497264186) q[6];
ry(2.943190425619611) q[7];
rz(-1.3432790224084066) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5010590088970077) q[0];
rz(-0.9104967172977649) q[0];
ry(-1.925927638590144) q[1];
rz(-1.402619598213767) q[1];
ry(-1.9280213493251204) q[2];
rz(-1.8987143798939234) q[2];
ry(0.9677152257469093) q[3];
rz(-0.7499278454863977) q[3];
ry(0.41874004738485215) q[4];
rz(-2.5080570915946816) q[4];
ry(1.9865329886949312) q[5];
rz(-2.2758667006270503) q[5];
ry(2.53654170883047) q[6];
rz(-0.9417616547189197) q[6];
ry(-1.6807622520924996) q[7];
rz(2.4069724555990253) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.847376035258018) q[0];
rz(2.2451060270274774) q[0];
ry(0.2398414117147455) q[1];
rz(0.2763460390345118) q[1];
ry(1.5250860137941897) q[2];
rz(-1.1741938537809118) q[2];
ry(-3.0039349150853005) q[3];
rz(-1.8429444188145774) q[3];
ry(-0.993618919328064) q[4];
rz(-3.022022658825054) q[4];
ry(0.7891165926887371) q[5];
rz(2.5465754622733323) q[5];
ry(-1.467228206448516) q[6];
rz(1.8753704644067337) q[6];
ry(-2.1371499735495565) q[7];
rz(-1.6286072742392987) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.4145400504904698) q[0];
rz(2.7654477329401326) q[0];
ry(-1.5417492861773365) q[1];
rz(1.3221953765487806) q[1];
ry(0.3583209969303253) q[2];
rz(-1.3507611352012088) q[2];
ry(-0.45267957700904704) q[3];
rz(0.5020830981156785) q[3];
ry(2.558106109567328) q[4];
rz(1.337767403080937) q[4];
ry(2.8180011004471344) q[5];
rz(1.2173889323752258) q[5];
ry(2.747639479470573) q[6];
rz(-0.5098383173908252) q[6];
ry(0.415956304131082) q[7];
rz(-1.5009580723093852) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.2935582562242756) q[0];
rz(-0.15741957627373804) q[0];
ry(0.47804899155452807) q[1];
rz(-2.0475595579249415) q[1];
ry(-0.9356451938893082) q[2];
rz(-1.6307876875023144) q[2];
ry(-0.17051888722960928) q[3];
rz(-0.8192137202388204) q[3];
ry(0.5792211578099898) q[4];
rz(0.5033378059989104) q[4];
ry(0.35932375080572826) q[5];
rz(2.997093856583761) q[5];
ry(2.3505884419118352) q[6];
rz(-2.250237502393757) q[6];
ry(0.40604078025574086) q[7];
rz(1.9940484597183312) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6066200118092189) q[0];
rz(-2.671124933304683) q[0];
ry(-2.849715684154643) q[1];
rz(1.348329599272487) q[1];
ry(1.4190933218342396) q[2];
rz(0.98315753808488) q[2];
ry(0.17650574371860372) q[3];
rz(3.078229216277902) q[3];
ry(1.902309184652107) q[4];
rz(0.03277908025217078) q[4];
ry(0.7815199268960532) q[5];
rz(0.970409075017833) q[5];
ry(-0.4334460961161252) q[6];
rz(2.994845407062157) q[6];
ry(2.8786476162776142) q[7];
rz(-2.8013248628589933) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.9537758435323749) q[0];
rz(-2.2308565169427466) q[0];
ry(-0.9265271502428298) q[1];
rz(0.15997235566772586) q[1];
ry(2.622438280623944) q[2];
rz(1.0535879331168843) q[2];
ry(1.2673980455182547) q[3];
rz(-0.15998428999726058) q[3];
ry(-1.0535223791944945) q[4];
rz(1.2100820088525313) q[4];
ry(-2.743794104953039) q[5];
rz(-1.3737363860957963) q[5];
ry(-0.1146331402628732) q[6];
rz(-1.7334992437792094) q[6];
ry(-3.035328099549611) q[7];
rz(0.6574334845311087) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5947327373804887) q[0];
rz(-0.6172064067129033) q[0];
ry(-0.9358431966691123) q[1];
rz(-0.4842671357593638) q[1];
ry(0.8879228055535001) q[2];
rz(2.032698698670343) q[2];
ry(1.9187471632206945) q[3];
rz(1.509807403361794) q[3];
ry(2.605873121239122) q[4];
rz(-2.3172734995734277) q[4];
ry(-2.7424839965698378) q[5];
rz(2.0802105439280125) q[5];
ry(-2.714516264777274) q[6];
rz(2.8300370288910286) q[6];
ry(-1.846418588324751) q[7];
rz(2.5512614858578524) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.47356631615659417) q[0];
rz(0.21305681004615626) q[0];
ry(-3.062246838735602) q[1];
rz(2.56462771744618) q[1];
ry(2.190352515732959) q[2];
rz(0.7504509224757046) q[2];
ry(0.6974235730759819) q[3];
rz(-1.6834891643143615) q[3];
ry(2.8943638835598127) q[4];
rz(2.484764277178727) q[4];
ry(0.24958940925563208) q[5];
rz(-0.42103803558454156) q[5];
ry(0.23202324139632857) q[6];
rz(0.3815464766142541) q[6];
ry(-1.1652427516040271) q[7];
rz(-3.0108089677130163) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.397117269812393) q[0];
rz(-1.5385356796247676) q[0];
ry(-0.6795113332934006) q[1];
rz(-1.6611044359113025) q[1];
ry(2.203557183590557) q[2];
rz(-0.33299824432508096) q[2];
ry(2.99978823557789) q[3];
rz(-0.21150823150365977) q[3];
ry(1.3666713119059484) q[4];
rz(-1.1363503261296173) q[4];
ry(1.415914189058178) q[5];
rz(1.0847528815488063) q[5];
ry(-1.6656180809167491) q[6];
rz(-1.8062797823605434) q[6];
ry(-1.269112124055722) q[7];
rz(-2.9085372682373674) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.839855582063614) q[0];
rz(1.9483702116636694) q[0];
ry(0.07400057762957761) q[1];
rz(-2.824122828098351) q[1];
ry(-1.481129298938077) q[2];
rz(-2.1809652428164545) q[2];
ry(0.771409092280697) q[3];
rz(-3.0574344992214137) q[3];
ry(2.1786076070540075) q[4];
rz(-2.7427590489449307) q[4];
ry(0.7767944161575278) q[5];
rz(0.18805447082119733) q[5];
ry(1.430766944845141) q[6];
rz(-2.0843842476691936) q[6];
ry(1.9045340074947) q[7];
rz(2.9341656201224424) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.851060457648753) q[0];
rz(-1.1347399826067859) q[0];
ry(-1.311574895937657) q[1];
rz(1.6839004883473234) q[1];
ry(1.443040467431703) q[2];
rz(2.578222602376648) q[2];
ry(0.8153037116927404) q[3];
rz(-2.3281975201772163) q[3];
ry(0.4962677513599986) q[4];
rz(1.2771259204390555) q[4];
ry(1.550638939113503) q[5];
rz(-1.424653294521086) q[5];
ry(-2.05524241389858) q[6];
rz(2.6045850511790323) q[6];
ry(2.029379700981657) q[7];
rz(2.0660337576971495) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.167060646803517) q[0];
rz(-1.8602601850629137) q[0];
ry(-2.475832422710999) q[1];
rz(2.5272581556195712) q[1];
ry(-2.778305625348485) q[2];
rz(1.5150390051991147) q[2];
ry(2.4196953064801714) q[3];
rz(-2.2509096372293818) q[3];
ry(2.164973881450316) q[4];
rz(1.6819043427752012) q[4];
ry(0.7107534983908643) q[5];
rz(1.89821734322563) q[5];
ry(-2.4339124561463823) q[6];
rz(2.1330296131559567) q[6];
ry(2.553710201670275) q[7];
rz(-1.6877791003980063) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.274745670217075) q[0];
rz(1.046380647240389) q[0];
ry(-2.431604131220153) q[1];
rz(2.6010176097821165) q[1];
ry(1.4646185402360201) q[2];
rz(1.129935403494916) q[2];
ry(-0.8296946608183139) q[3];
rz(-0.9077705618838791) q[3];
ry(-0.8024014279863493) q[4];
rz(2.6677017118523927) q[4];
ry(1.7965186314037414) q[5];
rz(-0.4253644523171758) q[5];
ry(1.8765456661523028) q[6];
rz(1.4503169913603866) q[6];
ry(2.5589694544950787) q[7];
rz(0.5503048963019461) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.0869490615391797) q[0];
rz(-2.4354227247198756) q[0];
ry(2.370670979409436) q[1];
rz(-2.993005217329403) q[1];
ry(0.7499423358303413) q[2];
rz(0.8135216930679565) q[2];
ry(2.7979344667615687) q[3];
rz(-2.024122002447206) q[3];
ry(0.5623530367832301) q[4];
rz(2.2676386478095214) q[4];
ry(2.4340814599296423) q[5];
rz(2.7290074500682238) q[5];
ry(1.0186616670134558) q[6];
rz(2.913573089994979) q[6];
ry(-2.178579174999196) q[7];
rz(0.32333354782170093) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.18908579444682694) q[0];
rz(0.42717622601933686) q[0];
ry(1.3377278697190227) q[1];
rz(2.767307587311268) q[1];
ry(-1.6679593712151022) q[2];
rz(0.7662507244314245) q[2];
ry(-0.9312214768914094) q[3];
rz(-0.9013718865270173) q[3];
ry(-2.257026380143961) q[4];
rz(2.837373469944298) q[4];
ry(-0.8028886086140563) q[5];
rz(1.6247592044783137) q[5];
ry(2.455018120708282) q[6];
rz(-1.0060853580688178) q[6];
ry(-0.5214752139753068) q[7];
rz(-0.9585455713072989) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.0400839430428297) q[0];
rz(2.789901842837902) q[0];
ry(0.8791637981912813) q[1];
rz(-3.09941273734808) q[1];
ry(-2.680324425415521) q[2];
rz(-2.415253970808839) q[2];
ry(2.844324084068975) q[3];
rz(2.7271703265945053) q[3];
ry(-0.7156060644584201) q[4];
rz(-3.012814536868574) q[4];
ry(-2.6525254109302336) q[5];
rz(2.389996916684781) q[5];
ry(-1.1506285087793495) q[6];
rz(-1.5982561949866758) q[6];
ry(1.3408430722824716) q[7];
rz(0.9539030769466453) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6004818232138738) q[0];
rz(2.0952407641593256) q[0];
ry(-0.4059054758155032) q[1];
rz(2.9832660161535287) q[1];
ry(0.03931753406414136) q[2];
rz(0.1556661810512543) q[2];
ry(-2.9231936684288162) q[3];
rz(0.3784302336133347) q[3];
ry(-2.4726316651420484) q[4];
rz(-1.4668759955821822) q[4];
ry(-0.5692103988367234) q[5];
rz(2.7373908858875464) q[5];
ry(-2.7342517235661186) q[6];
rz(2.8738921326167355) q[6];
ry(2.428712781659571) q[7];
rz(1.4757256398261267) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9185295057357249) q[0];
rz(2.709833459528551) q[0];
ry(2.3376452475928455) q[1];
rz(-1.785407092617624) q[1];
ry(-1.243863290851356) q[2];
rz(-2.0552019920087314) q[2];
ry(-0.47538356788775227) q[3];
rz(0.9044628722713774) q[3];
ry(-2.6422791323782344) q[4];
rz(2.486524538204746) q[4];
ry(-1.8631755674507067) q[5];
rz(-2.3058293840610173) q[5];
ry(1.9646803150940666) q[6];
rz(2.38090522259039) q[6];
ry(-2.547524246613168) q[7];
rz(3.111745550916245) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.704990480958069) q[0];
rz(0.6985290544531408) q[0];
ry(0.07531979891514505) q[1];
rz(1.4137821964296176) q[1];
ry(0.7669961067705682) q[2];
rz(-1.5548965812212436) q[2];
ry(0.8307036546151765) q[3];
rz(-2.8188009275986885) q[3];
ry(1.0706445989699676) q[4];
rz(0.8888797759516716) q[4];
ry(-1.388365357663669) q[5];
rz(1.2781084592618308) q[5];
ry(2.3794536857513027) q[6];
rz(-0.9172891261750885) q[6];
ry(-1.973625556678188) q[7];
rz(-2.6549606070843126) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9613630573901372) q[0];
rz(1.1104984989575142) q[0];
ry(1.6509232962449572) q[1];
rz(-3.0768581204524454) q[1];
ry(1.2730547697818633) q[2];
rz(2.750282334897094) q[2];
ry(-2.96971031832573) q[3];
rz(-2.8804702320777884) q[3];
ry(-2.611959466737088) q[4];
rz(1.0926686740433924) q[4];
ry(1.028618651137423) q[5];
rz(1.1508992708855734) q[5];
ry(1.63439551678583) q[6];
rz(2.4165059649095912) q[6];
ry(-1.3655063629764923) q[7];
rz(-2.520256912257949) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.371756151686042) q[0];
rz(2.0620540129330656) q[0];
ry(1.1169001806764083) q[1];
rz(2.1275695756649187) q[1];
ry(-1.3736246366030649) q[2];
rz(-0.14049743441298634) q[2];
ry(-2.4361846670701905) q[3];
rz(-0.652269833537499) q[3];
ry(-2.568956080858965) q[4];
rz(-2.319585299940881) q[4];
ry(1.5489368995536337) q[5];
rz(0.07746390420907634) q[5];
ry(0.22117054864649432) q[6];
rz(-0.5534541040693475) q[6];
ry(0.23779428575235165) q[7];
rz(-1.7632873788226329) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6862897177140423) q[0];
rz(-0.6526951885016583) q[0];
ry(2.1168572493383726) q[1];
rz(-1.2694888732131817) q[1];
ry(1.152824822557613) q[2];
rz(1.8919271817333387) q[2];
ry(-1.9438733565328925) q[3];
rz(-2.1075460003615456) q[3];
ry(2.457057863284811) q[4];
rz(-0.35933645725620417) q[4];
ry(1.2578809485551794) q[5];
rz(0.9344233104334788) q[5];
ry(0.9179684132896302) q[6];
rz(1.1242802856967868) q[6];
ry(-0.7668789131872424) q[7];
rz(-1.2643345920872668) q[7];