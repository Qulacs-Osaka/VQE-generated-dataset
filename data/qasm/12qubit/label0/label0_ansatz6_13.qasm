OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.923892707085139) q[0];
ry(0.45687842096881426) q[1];
cx q[0],q[1];
ry(-2.978705791976125) q[0];
ry(1.2936951398609304) q[1];
cx q[0],q[1];
ry(0.16901989130455938) q[1];
ry(0.25439399439953103) q[2];
cx q[1],q[2];
ry(-1.694515163050139) q[1];
ry(-2.4042905682561195) q[2];
cx q[1],q[2];
ry(1.9833767780282718) q[2];
ry(1.5708523241446963) q[3];
cx q[2],q[3];
ry(-0.8092826694062643) q[2];
ry(2.0839033781212756e-05) q[3];
cx q[2],q[3];
ry(1.4913431077276254) q[3];
ry(2.7471497370779416) q[4];
cx q[3],q[4];
ry(-2.867510417862481) q[3];
ry(-1.6797968529652894) q[4];
cx q[3],q[4];
ry(2.433269573368085) q[4];
ry(1.425722852585464) q[5];
cx q[4],q[5];
ry(-1.7552034424245784) q[4];
ry(-2.8923507095205014) q[5];
cx q[4],q[5];
ry(-2.5053206065669356) q[5];
ry(2.3824411837070314) q[6];
cx q[5],q[6];
ry(0.6351344739699359) q[5];
ry(-1.7414438591434704) q[6];
cx q[5],q[6];
ry(2.518983446395461) q[6];
ry(2.2465816771766605) q[7];
cx q[6],q[7];
ry(3.1414556629108694) q[6];
ry(3.1415874044173804) q[7];
cx q[6],q[7];
ry(-1.7873857251718235) q[7];
ry(1.3456159186578978) q[8];
cx q[7],q[8];
ry(-1.9645860397106547) q[7];
ry(-0.6583679995304864) q[8];
cx q[7],q[8];
ry(1.9058423744241704) q[8];
ry(0.7627870954298359) q[9];
cx q[8],q[9];
ry(0.07042506785491298) q[8];
ry(0.31386437593071304) q[9];
cx q[8],q[9];
ry(2.7262721158691816) q[9];
ry(1.0031956528439236) q[10];
cx q[9],q[10];
ry(2.294698174130489) q[9];
ry(0.664372022248722) q[10];
cx q[9],q[10];
ry(-3.0311091086562816) q[10];
ry(2.1974806985383597) q[11];
cx q[10],q[11];
ry(2.73788731451902) q[10];
ry(-0.1259535637314379) q[11];
cx q[10],q[11];
ry(2.724142264585699) q[0];
ry(-2.817618411022748) q[1];
cx q[0],q[1];
ry(-1.4996688555743063) q[0];
ry(-1.4758977435153158) q[1];
cx q[0],q[1];
ry(2.065238489319638) q[1];
ry(1.381896012225091) q[2];
cx q[1],q[2];
ry(-1.0606453414595463) q[1];
ry(0.7971958145666269) q[2];
cx q[1],q[2];
ry(2.694982417963471) q[2];
ry(0.3719704053371631) q[3];
cx q[2],q[3];
ry(1.3413344830188967) q[2];
ry(-2.9651577640017504) q[3];
cx q[2],q[3];
ry(2.8223185915023317) q[3];
ry(0.9573934751136388) q[4];
cx q[3],q[4];
ry(-2.98888803495053) q[3];
ry(3.1415378671972767) q[4];
cx q[3],q[4];
ry(-1.5614305858961304) q[4];
ry(0.4197380493739802) q[5];
cx q[4],q[5];
ry(0.0033385255003416407) q[4];
ry(-2.1222212630645583) q[5];
cx q[4],q[5];
ry(1.1827839411024517) q[5];
ry(-2.9654113994968423) q[6];
cx q[5],q[6];
ry(0.30956804475443345) q[5];
ry(-2.9918569346902375) q[6];
cx q[5],q[6];
ry(-1.3170024108295797) q[6];
ry(1.2256399678306797) q[7];
cx q[6],q[7];
ry(-2.743522160610984) q[6];
ry(0.0012513615067781814) q[7];
cx q[6],q[7];
ry(-1.6685758709426546) q[7];
ry(-2.2174576006768287) q[8];
cx q[7],q[8];
ry(-1.2181458858973864) q[7];
ry(0.737598180683265) q[8];
cx q[7],q[8];
ry(0.5775080448639978) q[8];
ry(-2.181318607097801) q[9];
cx q[8],q[9];
ry(-0.00010239404306222612) q[8];
ry(1.0545433130282333e-05) q[9];
cx q[8],q[9];
ry(1.082927571323288) q[9];
ry(1.405493169576019) q[10];
cx q[9],q[10];
ry(-0.39726783780849184) q[9];
ry(2.023782501455671) q[10];
cx q[9],q[10];
ry(0.7406097243489642) q[10];
ry(-1.9759471945207132) q[11];
cx q[10],q[11];
ry(1.3431526078233376) q[10];
ry(0.24793269240803717) q[11];
cx q[10],q[11];
ry(1.55757652638984) q[0];
ry(0.17436179953221093) q[1];
cx q[0],q[1];
ry(-0.808121858201301) q[0];
ry(0.7597113412745489) q[1];
cx q[0],q[1];
ry(2.3533900727315933) q[1];
ry(1.533727654425954) q[2];
cx q[1],q[2];
ry(3.0427750337604964) q[1];
ry(-2.1768434381754718) q[2];
cx q[1],q[2];
ry(-1.6609231705359253) q[2];
ry(2.1737393839171726) q[3];
cx q[2],q[3];
ry(-1.519540008514885) q[2];
ry(-0.2074454488234423) q[3];
cx q[2],q[3];
ry(2.581782277268385) q[3];
ry(-1.2718649883536675) q[4];
cx q[3],q[4];
ry(0.003676833741179293) q[3];
ry(-3.1386712192216057) q[4];
cx q[3],q[4];
ry(-1.5276672066242372) q[4];
ry(2.382777268574477) q[5];
cx q[4],q[5];
ry(1.2603089702652677) q[4];
ry(0.5956310060232461) q[5];
cx q[4],q[5];
ry(0.6880230551128831) q[5];
ry(-0.1034840531723864) q[6];
cx q[5],q[6];
ry(-0.8304125929013024) q[5];
ry(-1.5761235726246798) q[6];
cx q[5],q[6];
ry(-0.524056395010815) q[6];
ry(-0.27931788682439057) q[7];
cx q[6],q[7];
ry(-3.1213266486683544) q[6];
ry(0.3877185051535301) q[7];
cx q[6],q[7];
ry(-1.5908020971892822) q[7];
ry(-1.169283531038297) q[8];
cx q[7],q[8];
ry(2.207873243506915) q[7];
ry(-0.5229989850975518) q[8];
cx q[7],q[8];
ry(-1.639692482666908) q[8];
ry(0.49617182946287725) q[9];
cx q[8],q[9];
ry(-0.3635796204199968) q[8];
ry(-0.4423180581713373) q[9];
cx q[8],q[9];
ry(-2.7271332275942983) q[9];
ry(-0.02519532105143216) q[10];
cx q[9],q[10];
ry(1.9078215145000936) q[9];
ry(1.8406598223536839) q[10];
cx q[9],q[10];
ry(-2.5305548538371743) q[10];
ry(1.2695151953924693) q[11];
cx q[10],q[11];
ry(-2.7466464095636516) q[10];
ry(2.675576480724112) q[11];
cx q[10],q[11];
ry(0.2973747079879816) q[0];
ry(-1.5718270008703896) q[1];
cx q[0],q[1];
ry(-0.9420101061033059) q[0];
ry(2.7744055190544508) q[1];
cx q[0],q[1];
ry(0.06784088849293468) q[1];
ry(-1.415060635325422) q[2];
cx q[1],q[2];
ry(1.4306016391555572) q[1];
ry(-1.5404463801999904) q[2];
cx q[1],q[2];
ry(1.093184621787756) q[2];
ry(1.166272324573969) q[3];
cx q[2],q[3];
ry(1.7914100884281403) q[2];
ry(-3.1324745923829385) q[3];
cx q[2],q[3];
ry(-2.535615077883894) q[3];
ry(0.3379183147024492) q[4];
cx q[3],q[4];
ry(-1.40112219791839) q[3];
ry(0.6398287884100436) q[4];
cx q[3],q[4];
ry(-2.629182772912219) q[4];
ry(0.6313402653565001) q[5];
cx q[4],q[5];
ry(-3.054705269792975) q[4];
ry(3.055497771505887) q[5];
cx q[4],q[5];
ry(2.8585061947736867) q[5];
ry(0.917580709338325) q[6];
cx q[5],q[6];
ry(3.070991677944551) q[5];
ry(-0.12687741435572075) q[6];
cx q[5],q[6];
ry(2.7287313622921308) q[6];
ry(0.15081612029752733) q[7];
cx q[6],q[7];
ry(-1.0374516471880388) q[6];
ry(-1.0804891015607152) q[7];
cx q[6],q[7];
ry(-1.8313774339016593) q[7];
ry(-0.7016656622314111) q[8];
cx q[7],q[8];
ry(0.4963818082835474) q[7];
ry(0.5232030728235374) q[8];
cx q[7],q[8];
ry(-2.530599383123613) q[8];
ry(-1.7660586446191964) q[9];
cx q[8],q[9];
ry(-0.021779645308672307) q[8];
ry(1.2554941282235201) q[9];
cx q[8],q[9];
ry(-1.5007522532205682) q[9];
ry(-2.09108275168541) q[10];
cx q[9],q[10];
ry(-2.0386771171939575) q[9];
ry(3.1259682474109454) q[10];
cx q[9],q[10];
ry(0.14838238935912956) q[10];
ry(1.5741943202513955) q[11];
cx q[10],q[11];
ry(-0.5562044290224843) q[10];
ry(0.36606982167376056) q[11];
cx q[10],q[11];
ry(1.864922514482511) q[0];
ry(2.0580488619150543) q[1];
cx q[0],q[1];
ry(0.45750191967120113) q[0];
ry(-1.3102780471894233) q[1];
cx q[0],q[1];
ry(2.512780551858077) q[1];
ry(2.5018705838420305) q[2];
cx q[1],q[2];
ry(-0.751095617901588) q[1];
ry(-3.0744905576587573) q[2];
cx q[1],q[2];
ry(-2.6813348238795505) q[2];
ry(-0.9228821869578772) q[3];
cx q[2],q[3];
ry(2.139878522544132) q[2];
ry(-2.048588173137723) q[3];
cx q[2],q[3];
ry(0.28843437136613875) q[3];
ry(3.129230444173279) q[4];
cx q[3],q[4];
ry(2.658981900151605) q[3];
ry(-3.1392018766175656) q[4];
cx q[3],q[4];
ry(1.8401670512456867) q[4];
ry(0.4860835051559736) q[5];
cx q[4],q[5];
ry(-3.0304721419022327) q[4];
ry(-2.7076916137976834) q[5];
cx q[4],q[5];
ry(0.5457845491105582) q[5];
ry(-2.4905513092601845) q[6];
cx q[5],q[6];
ry(0.3373160870841626) q[5];
ry(-0.4537101519748781) q[6];
cx q[5],q[6];
ry(1.6772381661273963) q[6];
ry(-1.6057565278705344) q[7];
cx q[6],q[7];
ry(0.001867030579729926) q[6];
ry(0.0006643758922555421) q[7];
cx q[6],q[7];
ry(-2.0220352426133656) q[7];
ry(-2.948978984618838) q[8];
cx q[7],q[8];
ry(3.0934729138169437) q[7];
ry(3.130834303936238) q[8];
cx q[7],q[8];
ry(0.16751843855127294) q[8];
ry(0.0046004946681383885) q[9];
cx q[8],q[9];
ry(-3.134519021445174) q[8];
ry(-1.3157908123526978) q[9];
cx q[8],q[9];
ry(-2.638703413330709) q[9];
ry(2.081125766110485) q[10];
cx q[9],q[10];
ry(1.645444430353935) q[9];
ry(-1.3823185466340826) q[10];
cx q[9],q[10];
ry(-2.5044501267625856) q[10];
ry(-2.6902249776720044) q[11];
cx q[10],q[11];
ry(-0.27950223595914725) q[10];
ry(3.0961192587767123) q[11];
cx q[10],q[11];
ry(1.1522143140929542) q[0];
ry(1.1849418159243832) q[1];
cx q[0],q[1];
ry(-0.4978815148395146) q[0];
ry(2.9480991445210494) q[1];
cx q[0],q[1];
ry(-2.3358498789559117) q[1];
ry(-0.18973109719612063) q[2];
cx q[1],q[2];
ry(-3.1264476170893176) q[1];
ry(0.156353570464975) q[2];
cx q[1],q[2];
ry(-0.9532723224348248) q[2];
ry(-2.4643568251953325) q[3];
cx q[2],q[3];
ry(-2.1186477913974873) q[2];
ry(-1.3238447047673745) q[3];
cx q[2],q[3];
ry(1.346101645591121) q[3];
ry(0.2029439418509092) q[4];
cx q[3],q[4];
ry(1.6640303339044096) q[3];
ry(-0.8098392380951367) q[4];
cx q[3],q[4];
ry(-0.882970931936315) q[4];
ry(-0.16268144656051245) q[5];
cx q[4],q[5];
ry(2.7843693263490765) q[4];
ry(2.991043337296056) q[5];
cx q[4],q[5];
ry(3.0811942838721977) q[5];
ry(-0.6549934169315774) q[6];
cx q[5],q[6];
ry(-0.9223701831102691) q[5];
ry(0.03016271249090514) q[6];
cx q[5],q[6];
ry(-0.3924029874164905) q[6];
ry(0.7097778301340103) q[7];
cx q[6],q[7];
ry(3.1378040935855696) q[6];
ry(-3.1369527661048) q[7];
cx q[6],q[7];
ry(0.2919949744600486) q[7];
ry(-0.8621418146544315) q[8];
cx q[7],q[8];
ry(0.46203688878143684) q[7];
ry(-0.1571538066993378) q[8];
cx q[7],q[8];
ry(2.9801310791062514) q[8];
ry(0.8381158689136294) q[9];
cx q[8],q[9];
ry(3.110417872656059) q[8];
ry(-0.008110904715150193) q[9];
cx q[8],q[9];
ry(1.0337116477188628) q[9];
ry(-0.14570293825076677) q[10];
cx q[9],q[10];
ry(1.2430094347676208) q[9];
ry(-2.4258643699275977) q[10];
cx q[9],q[10];
ry(-2.3622757667141627) q[10];
ry(0.22448418621821897) q[11];
cx q[10],q[11];
ry(0.854421107936162) q[10];
ry(-1.7496328619962531) q[11];
cx q[10],q[11];
ry(0.11117689042483202) q[0];
ry(1.6684261891116068) q[1];
cx q[0],q[1];
ry(1.4154159105154245) q[0];
ry(2.8287728156492293) q[1];
cx q[0],q[1];
ry(1.492732500224288) q[1];
ry(-2.6147361446182456) q[2];
cx q[1],q[2];
ry(1.7199948920536066) q[1];
ry(0.15204179513294486) q[2];
cx q[1],q[2];
ry(0.785915902790296) q[2];
ry(-0.554437457940285) q[3];
cx q[2],q[3];
ry(-2.8414108332251704) q[2];
ry(2.9869224924810913) q[3];
cx q[2],q[3];
ry(2.369828962825192) q[3];
ry(-1.4251632701164045) q[4];
cx q[3],q[4];
ry(0.3232801984164666) q[3];
ry(2.991478216853275) q[4];
cx q[3],q[4];
ry(1.7093630026112772) q[4];
ry(0.6555437290438837) q[5];
cx q[4],q[5];
ry(-2.477019312044028) q[4];
ry(-0.10739081138374562) q[5];
cx q[4],q[5];
ry(-2.3660992391262092) q[5];
ry(-1.9598006570187678) q[6];
cx q[5],q[6];
ry(-0.0625774055699031) q[5];
ry(-3.0186196666924054) q[6];
cx q[5],q[6];
ry(-0.11997185037346014) q[6];
ry(-1.7119882398310704) q[7];
cx q[6],q[7];
ry(3.1045490861023586) q[6];
ry(0.001328129889915708) q[7];
cx q[6],q[7];
ry(-1.7765056729091933) q[7];
ry(-2.4709920406233232) q[8];
cx q[7],q[8];
ry(2.783174386997761) q[7];
ry(-0.3323631003283558) q[8];
cx q[7],q[8];
ry(-1.6328951039638626) q[8];
ry(-0.9790779929886042) q[9];
cx q[8],q[9];
ry(-0.02848586081735416) q[8];
ry(0.02783118807534503) q[9];
cx q[8],q[9];
ry(-0.21570793124360177) q[9];
ry(-0.9265874109301331) q[10];
cx q[9],q[10];
ry(1.8143147061470957) q[9];
ry(3.026518878188516) q[10];
cx q[9],q[10];
ry(0.29144230658018416) q[10];
ry(2.177539545728118) q[11];
cx q[10],q[11];
ry(-1.1388503256551255) q[10];
ry(-3.054056404924992) q[11];
cx q[10],q[11];
ry(-1.524861606207709) q[0];
ry(-2.80465148322328) q[1];
cx q[0],q[1];
ry(-0.5097304889726841) q[0];
ry(2.0886493999896176) q[1];
cx q[0],q[1];
ry(1.1515424612077876) q[1];
ry(2.2004007592474446) q[2];
cx q[1],q[2];
ry(-1.4619881108571606) q[1];
ry(1.4351591259926257) q[2];
cx q[1],q[2];
ry(-1.3471156316411939) q[2];
ry(1.7075571161386425) q[3];
cx q[2],q[3];
ry(1.9536278883995604) q[2];
ry(-0.17244265209242926) q[3];
cx q[2],q[3];
ry(-1.4754021682856004) q[3];
ry(0.2495576783473492) q[4];
cx q[3],q[4];
ry(3.1026634238088016) q[3];
ry(-2.1056054290782047) q[4];
cx q[3],q[4];
ry(-3.048876502912528) q[4];
ry(2.749719887843293) q[5];
cx q[4],q[5];
ry(-3.0797931829993628) q[4];
ry(-0.04619945676930717) q[5];
cx q[4],q[5];
ry(-0.8670310604075224) q[5];
ry(-0.8139012272930121) q[6];
cx q[5],q[6];
ry(3.1390771847949015) q[5];
ry(2.964634821346992) q[6];
cx q[5],q[6];
ry(1.765229530240579) q[6];
ry(0.534236970890441) q[7];
cx q[6],q[7];
ry(3.1040494481392877) q[6];
ry(-0.000733870557466913) q[7];
cx q[6],q[7];
ry(-0.5601231008082603) q[7];
ry(1.383670181489539) q[8];
cx q[7],q[8];
ry(-0.540188189402385) q[7];
ry(-1.7856629961475334) q[8];
cx q[7],q[8];
ry(-0.7327886225263596) q[8];
ry(1.1141666366747556) q[9];
cx q[8],q[9];
ry(3.1167351533742846) q[8];
ry(-0.024083922062424077) q[9];
cx q[8],q[9];
ry(2.265442990870329) q[9];
ry(0.36783125714356846) q[10];
cx q[9],q[10];
ry(2.693319342405181) q[9];
ry(-0.5177026597288199) q[10];
cx q[9],q[10];
ry(-2.374789587718128) q[10];
ry(0.8368862715961762) q[11];
cx q[10],q[11];
ry(-1.2401483552585402) q[10];
ry(1.8175869663611324) q[11];
cx q[10],q[11];
ry(-3.0306370433220406) q[0];
ry(-2.0355780899570415) q[1];
cx q[0],q[1];
ry(-3.1092089976700428) q[0];
ry(1.6865493822588071) q[1];
cx q[0],q[1];
ry(-2.5471867277744025) q[1];
ry(1.1909613822154466) q[2];
cx q[1],q[2];
ry(-0.406360546552583) q[1];
ry(-0.6106643561815543) q[2];
cx q[1],q[2];
ry(-1.6163660041659176) q[2];
ry(-2.4842260082529792) q[3];
cx q[2],q[3];
ry(2.2090429807536376) q[2];
ry(-1.6434539701584543) q[3];
cx q[2],q[3];
ry(-1.213504103955522) q[3];
ry(-0.4530888437266718) q[4];
cx q[3],q[4];
ry(-0.01909896934859656) q[3];
ry(-2.882921838430141) q[4];
cx q[3],q[4];
ry(1.3416776346159187) q[4];
ry(-0.18704999547174683) q[5];
cx q[4],q[5];
ry(-2.2301581433576576) q[4];
ry(-0.13625635269004555) q[5];
cx q[4],q[5];
ry(-1.930187669182665) q[5];
ry(3.069829317091269) q[6];
cx q[5],q[6];
ry(-0.01111764485176214) q[5];
ry(-0.0034854257138574596) q[6];
cx q[5],q[6];
ry(0.8837433987040351) q[6];
ry(0.5046398172541268) q[7];
cx q[6],q[7];
ry(3.089472767675074) q[6];
ry(-3.140799364516149) q[7];
cx q[6],q[7];
ry(2.2929390500563946) q[7];
ry(-1.498941811419332) q[8];
cx q[7],q[8];
ry(-2.667616109932704) q[7];
ry(-1.5609676571252233) q[8];
cx q[7],q[8];
ry(1.2500114506747089) q[8];
ry(-2.1268700140285066) q[9];
cx q[8],q[9];
ry(-0.03664320803840673) q[8];
ry(0.9830378748809451) q[9];
cx q[8],q[9];
ry(1.8042508069810586) q[9];
ry(2.8164841118083914) q[10];
cx q[9],q[10];
ry(-1.3340580556364845) q[9];
ry(0.021334841660785035) q[10];
cx q[9],q[10];
ry(-1.3565379865678011) q[10];
ry(2.616551825634512) q[11];
cx q[10],q[11];
ry(0.8001392249397875) q[10];
ry(1.2512400538038015) q[11];
cx q[10],q[11];
ry(0.13416602663938892) q[0];
ry(-1.997335566752572) q[1];
cx q[0],q[1];
ry(-0.007794478730473223) q[0];
ry(-2.1462272003197715) q[1];
cx q[0],q[1];
ry(0.7065275275244568) q[1];
ry(0.008168536370133593) q[2];
cx q[1],q[2];
ry(1.9620007576174876) q[1];
ry(0.18271569641708965) q[2];
cx q[1],q[2];
ry(0.5162224808854079) q[2];
ry(2.727744251188108) q[3];
cx q[2],q[3];
ry(-1.5682362156478975) q[2];
ry(-1.1920597656724423) q[3];
cx q[2],q[3];
ry(-1.3025379161708805) q[3];
ry(-2.6912594575659665) q[4];
cx q[3],q[4];
ry(1.397880012373112) q[3];
ry(2.8497988856793763) q[4];
cx q[3],q[4];
ry(-1.9132709465718105) q[4];
ry(-0.8331586677632874) q[5];
cx q[4],q[5];
ry(-3.073292023462629) q[4];
ry(0.010124728250755943) q[5];
cx q[4],q[5];
ry(-0.6160519041802313) q[5];
ry(0.10981043692992822) q[6];
cx q[5],q[6];
ry(-3.1377673269352577) q[5];
ry(-0.17307785158603384) q[6];
cx q[5],q[6];
ry(0.3193033494601196) q[6];
ry(0.44868442335377134) q[7];
cx q[6],q[7];
ry(0.29592836171936837) q[6];
ry(2.7892047927752515) q[7];
cx q[6],q[7];
ry(-2.883716305295125) q[7];
ry(-1.0292695390820272) q[8];
cx q[7],q[8];
ry(0.14752297873292933) q[7];
ry(-0.004800063014026358) q[8];
cx q[7],q[8];
ry(0.23376814008010705) q[8];
ry(-2.46458119960336) q[9];
cx q[8],q[9];
ry(2.2443348548589857) q[8];
ry(2.7014423306282547) q[9];
cx q[8],q[9];
ry(-0.25982509225299333) q[9];
ry(-2.553073650818588) q[10];
cx q[9],q[10];
ry(-2.9142021210652) q[9];
ry(-1.8017701115057527) q[10];
cx q[9],q[10];
ry(0.19283080025055305) q[10];
ry(1.618476055723092) q[11];
cx q[10],q[11];
ry(0.7872241857395151) q[10];
ry(-0.17754604572538213) q[11];
cx q[10],q[11];
ry(-0.06492645567426407) q[0];
ry(-2.093579820729116) q[1];
cx q[0],q[1];
ry(-3.136706078160079) q[0];
ry(1.9612547195847263) q[1];
cx q[0],q[1];
ry(-2.6169646733434972) q[1];
ry(-0.748404120411414) q[2];
cx q[1],q[2];
ry(-0.8985966932084449) q[1];
ry(1.4217807594850045) q[2];
cx q[1],q[2];
ry(-3.105286488549155) q[2];
ry(-2.8114614981468056) q[3];
cx q[2],q[3];
ry(0.026981980933096675) q[2];
ry(-1.8230167296492141) q[3];
cx q[2],q[3];
ry(-1.3995731639559652) q[3];
ry(-1.5615690385157146) q[4];
cx q[3],q[4];
ry(1.9563518024624909) q[3];
ry(1.0584897365243435) q[4];
cx q[3],q[4];
ry(-2.6533421639049326) q[4];
ry(1.634844839712013) q[5];
cx q[4],q[5];
ry(1.799131745191504) q[4];
ry(-0.0007593472741929119) q[5];
cx q[4],q[5];
ry(-2.499992741355525) q[5];
ry(2.811742206952593) q[6];
cx q[5],q[6];
ry(-0.12551675765662118) q[5];
ry(-1.6266766877829075) q[6];
cx q[5],q[6];
ry(-2.7615834240951025) q[6];
ry(1.471254258403432) q[7];
cx q[6],q[7];
ry(0.12359291819072293) q[6];
ry(3.069462369368165) q[7];
cx q[6],q[7];
ry(-0.5728973398214849) q[7];
ry(-2.2228592227445967) q[8];
cx q[7],q[8];
ry(-3.1390773095120053) q[7];
ry(0.07618851472459909) q[8];
cx q[7],q[8];
ry(-2.1487505283450163) q[8];
ry(-2.4066525313261837) q[9];
cx q[8],q[9];
ry(-2.2920995867845133) q[8];
ry(3.094061116641019) q[9];
cx q[8],q[9];
ry(1.0669030139523186) q[9];
ry(-2.9377417881843533) q[10];
cx q[9],q[10];
ry(2.9249684220402816) q[9];
ry(1.2484267132578877) q[10];
cx q[9],q[10];
ry(-1.3982952772148973) q[10];
ry(0.5115023863468044) q[11];
cx q[10],q[11];
ry(-2.4938555684646317) q[10];
ry(3.086670318942798) q[11];
cx q[10],q[11];
ry(1.845459484358634) q[0];
ry(-1.6050983023121057) q[1];
cx q[0],q[1];
ry(0.07391466959974391) q[0];
ry(-1.1379635606055207) q[1];
cx q[0],q[1];
ry(-1.5522687754694162) q[1];
ry(0.5034561244669512) q[2];
cx q[1],q[2];
ry(-2.7908605463429126) q[1];
ry(-3.0920863428053478) q[2];
cx q[1],q[2];
ry(-1.9005000338980924) q[2];
ry(1.219082966326983) q[3];
cx q[2],q[3];
ry(-0.3595883321026773) q[2];
ry(1.1919639628736505) q[3];
cx q[2],q[3];
ry(2.7908191984478457) q[3];
ry(0.023352141091305022) q[4];
cx q[3],q[4];
ry(-2.5769978815923724) q[3];
ry(-2.8334578854067605) q[4];
cx q[3],q[4];
ry(0.7622953275585429) q[4];
ry(0.0532852266937951) q[5];
cx q[4],q[5];
ry(0.0021353128187193474) q[4];
ry(-0.00019545975044898967) q[5];
cx q[4],q[5];
ry(-2.996357959856793) q[5];
ry(2.918010973070282) q[6];
cx q[5],q[6];
ry(3.0368654392611534) q[5];
ry(-1.903578377741664) q[6];
cx q[5],q[6];
ry(1.0310737847076137) q[6];
ry(-0.016798728654547013) q[7];
cx q[6],q[7];
ry(-0.8385548122678683) q[6];
ry(3.10697938274238) q[7];
cx q[6],q[7];
ry(-1.157779097214837) q[7];
ry(2.5555451340153366) q[8];
cx q[7],q[8];
ry(-3.1364339164466277) q[7];
ry(-1.1702468502014822) q[8];
cx q[7],q[8];
ry(2.929840110313809) q[8];
ry(0.26871019469391494) q[9];
cx q[8],q[9];
ry(-2.8932441538185025) q[8];
ry(0.0008696963639795996) q[9];
cx q[8],q[9];
ry(-2.9957161262561423) q[9];
ry(-0.510518475195591) q[10];
cx q[9],q[10];
ry(-0.8693556005877037) q[9];
ry(-2.511949881781149) q[10];
cx q[9],q[10];
ry(1.769472873057205) q[10];
ry(-1.1770378094611103) q[11];
cx q[10],q[11];
ry(1.3143097126119383) q[10];
ry(1.9785966824422654) q[11];
cx q[10],q[11];
ry(-1.3686443171140927) q[0];
ry(-1.5079783875199535) q[1];
cx q[0],q[1];
ry(-3.1109640771525897) q[0];
ry(-1.9784288432001) q[1];
cx q[0],q[1];
ry(-1.9459157817718744) q[1];
ry(1.2279794112229148) q[2];
cx q[1],q[2];
ry(-2.462718996862784) q[1];
ry(-0.028462976812009266) q[2];
cx q[1],q[2];
ry(2.4192040587673147) q[2];
ry(-2.1257975251966625) q[3];
cx q[2],q[3];
ry(-0.03663371521943582) q[2];
ry(-1.6529874814367025) q[3];
cx q[2],q[3];
ry(-1.0084636515597172) q[3];
ry(-0.3568841313798493) q[4];
cx q[3],q[4];
ry(-0.4930964347044907) q[3];
ry(2.729261403276098) q[4];
cx q[3],q[4];
ry(-1.840920992116108) q[4];
ry(1.9038902656649) q[5];
cx q[4],q[5];
ry(-2.3534606349474196) q[4];
ry(3.1393212409902995) q[5];
cx q[4],q[5];
ry(0.31905628877464665) q[5];
ry(2.1000589898897384) q[6];
cx q[5],q[6];
ry(-0.029379272598172967) q[5];
ry(3.124613605660074) q[6];
cx q[5],q[6];
ry(-0.653250576273809) q[6];
ry(3.1284920680182373) q[7];
cx q[6],q[7];
ry(-3.132668940785222) q[6];
ry(0.031633213772864544) q[7];
cx q[6],q[7];
ry(2.4655261537386868) q[7];
ry(-0.17868571401548472) q[8];
cx q[7],q[8];
ry(-2.47801322640188) q[7];
ry(-1.9503413607747253) q[8];
cx q[7],q[8];
ry(0.6140949287371733) q[8];
ry(2.8339603668153472) q[9];
cx q[8],q[9];
ry(-0.7471426181867552) q[8];
ry(-0.8969903006873544) q[9];
cx q[8],q[9];
ry(3.1116385551785837) q[9];
ry(0.6563700246378583) q[10];
cx q[9],q[10];
ry(-3.108613407621938) q[9];
ry(-3.139654916026371) q[10];
cx q[9],q[10];
ry(0.437242273200817) q[10];
ry(-1.2714782614688536) q[11];
cx q[10],q[11];
ry(2.1482599330460124) q[10];
ry(-0.6939877297983178) q[11];
cx q[10],q[11];
ry(-3.0417479899433792) q[0];
ry(1.8339647641303358) q[1];
cx q[0],q[1];
ry(1.6137960019243618) q[0];
ry(-2.8800180354309646) q[1];
cx q[0],q[1];
ry(2.7275836635011785) q[1];
ry(-0.8865225197228106) q[2];
cx q[1],q[2];
ry(0.07203138710128289) q[1];
ry(3.0119835045382635) q[2];
cx q[1],q[2];
ry(-1.2893171071852543) q[2];
ry(1.2861805756349556) q[3];
cx q[2],q[3];
ry(2.8686320123696034) q[2];
ry(-0.4100720257184781) q[3];
cx q[2],q[3];
ry(-0.6242896654369017) q[3];
ry(-1.473724565137764) q[4];
cx q[3],q[4];
ry(-3.1305221877694898) q[3];
ry(-2.6684189747493137) q[4];
cx q[3],q[4];
ry(-0.8037146880158543) q[4];
ry(-2.6935094188703506) q[5];
cx q[4],q[5];
ry(-2.04135340116862) q[4];
ry(3.1403958492604502) q[5];
cx q[4],q[5];
ry(3.080327825224036) q[5];
ry(-2.519233718924143) q[6];
cx q[5],q[6];
ry(-2.198271353213566) q[5];
ry(2.1193378145013386) q[6];
cx q[5],q[6];
ry(-1.4898943554766912) q[6];
ry(1.6309615487225093) q[7];
cx q[6],q[7];
ry(-3.1245129002491407) q[6];
ry(-3.1385840971580956) q[7];
cx q[6],q[7];
ry(-2.53013304330594) q[7];
ry(-0.584252840148416) q[8];
cx q[7],q[8];
ry(-0.003915362625866086) q[7];
ry(0.00662949702579889) q[8];
cx q[7],q[8];
ry(-3.0407375139370565) q[8];
ry(-0.31449157031224356) q[9];
cx q[8],q[9];
ry(2.469904191628819) q[8];
ry(0.734032256806783) q[9];
cx q[8],q[9];
ry(2.73095244897915) q[9];
ry(-1.3482611773049253) q[10];
cx q[9],q[10];
ry(-0.6731732393319295) q[9];
ry(-0.8873392455144042) q[10];
cx q[9],q[10];
ry(-1.3041562580024415) q[10];
ry(-1.1362656478225535) q[11];
cx q[10],q[11];
ry(0.7542579674301573) q[10];
ry(0.38510353345961507) q[11];
cx q[10],q[11];
ry(2.5798726299930625) q[0];
ry(0.6715096765690983) q[1];
cx q[0],q[1];
ry(-0.3203395230141002) q[0];
ry(0.12137232494043336) q[1];
cx q[0],q[1];
ry(2.2044596887419567) q[1];
ry(2.3512098142170275) q[2];
cx q[1],q[2];
ry(0.02351014844228069) q[1];
ry(3.01767875464246) q[2];
cx q[1],q[2];
ry(2.3965344600307468) q[2];
ry(-0.3035134125828241) q[3];
cx q[2],q[3];
ry(-2.3345559991365246) q[2];
ry(3.0246989658514196) q[3];
cx q[2],q[3];
ry(-1.858098109749651) q[3];
ry(0.46539073761934535) q[4];
cx q[3],q[4];
ry(0.028411200183155655) q[3];
ry(0.49805166059512374) q[4];
cx q[3],q[4];
ry(-1.393669091344523) q[4];
ry(0.6823626220898563) q[5];
cx q[4],q[5];
ry(-0.013940798026641945) q[4];
ry(0.0004217510297834792) q[5];
cx q[4],q[5];
ry(2.4475193768477186) q[5];
ry(-1.3441605536226628) q[6];
cx q[5],q[6];
ry(2.3414888677653996) q[5];
ry(0.8657885080213008) q[6];
cx q[5],q[6];
ry(-0.6666982514498176) q[6];
ry(0.032349972723952215) q[7];
cx q[6],q[7];
ry(-0.21647845469881588) q[6];
ry(0.01094018469697122) q[7];
cx q[6],q[7];
ry(2.024525253326775) q[7];
ry(-0.5431755092998785) q[8];
cx q[7],q[8];
ry(-1.302021309684502) q[7];
ry(-3.030743610179569) q[8];
cx q[7],q[8];
ry(2.7895620338049265) q[8];
ry(2.952139642567467) q[9];
cx q[8],q[9];
ry(-0.007590786434756858) q[8];
ry(-0.004161951096713246) q[9];
cx q[8],q[9];
ry(1.4119791607215726) q[9];
ry(1.047028182970477) q[10];
cx q[9],q[10];
ry(1.8190428708642834) q[9];
ry(-0.7187704956356198) q[10];
cx q[9],q[10];
ry(0.149584921740993) q[10];
ry(2.164564991664709) q[11];
cx q[10],q[11];
ry(-1.2879672257994466) q[10];
ry(0.4725098954172539) q[11];
cx q[10],q[11];
ry(-1.311637030108806) q[0];
ry(-2.7305328126910426) q[1];
cx q[0],q[1];
ry(1.027316852602185) q[0];
ry(-1.8518789997983132) q[1];
cx q[0],q[1];
ry(-2.8110272807248893) q[1];
ry(0.6019490358205868) q[2];
cx q[1],q[2];
ry(-0.015634991795311137) q[1];
ry(-0.16141819426568452) q[2];
cx q[1],q[2];
ry(-0.9926131934911471) q[2];
ry(-2.0150895233794826) q[3];
cx q[2],q[3];
ry(-2.62593592957625) q[2];
ry(0.5109477904772368) q[3];
cx q[2],q[3];
ry(-0.268893877820602) q[3];
ry(2.7987945612982683) q[4];
cx q[3],q[4];
ry(0.11024990358433139) q[3];
ry(1.2152068547154584) q[4];
cx q[3],q[4];
ry(-1.6034647887298572) q[4];
ry(-3.0324069747721634) q[5];
cx q[4],q[5];
ry(-3.119826543609625) q[4];
ry(0.008311705520812396) q[5];
cx q[4],q[5];
ry(1.8196400490399727) q[5];
ry(-0.89980332419463) q[6];
cx q[5],q[6];
ry(-0.07053195814449344) q[5];
ry(2.956494424838134) q[6];
cx q[5],q[6];
ry(2.4235929342683558) q[6];
ry(-1.2789796849873962) q[7];
cx q[6],q[7];
ry(3.0950325618943455) q[6];
ry(-2.631355199611396) q[7];
cx q[6],q[7];
ry(-1.3381341191093794) q[7];
ry(0.026698027293688753) q[8];
cx q[7],q[8];
ry(0.5679439655248352) q[7];
ry(-2.991941684246515) q[8];
cx q[7],q[8];
ry(-2.638643783658421) q[8];
ry(1.1943512474141496) q[9];
cx q[8],q[9];
ry(-0.018045255008133587) q[8];
ry(3.1371363255326608) q[9];
cx q[8],q[9];
ry(-0.23949992943028314) q[9];
ry(-0.17362655747434985) q[10];
cx q[9],q[10];
ry(-0.5749715532228947) q[9];
ry(-0.48942923469146876) q[10];
cx q[9],q[10];
ry(-2.775751799346192) q[10];
ry(-0.09353280347750816) q[11];
cx q[10],q[11];
ry(2.2833378979870416) q[10];
ry(0.6710547455445798) q[11];
cx q[10],q[11];
ry(2.6960435765950805) q[0];
ry(0.5978579402937636) q[1];
ry(2.078663212354013) q[2];
ry(-3.1218883941901634) q[3];
ry(0.5939029286799169) q[4];
ry(0.3518445081195583) q[5];
ry(-0.03659068602444919) q[6];
ry(-3.036237147975587) q[7];
ry(-1.507341144958469) q[8];
ry(-2.601959124159993) q[9];
ry(-2.544166284937056) q[10];
ry(0.05816286607795629) q[11];