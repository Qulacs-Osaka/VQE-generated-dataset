OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.2228787645278523) q[0];
ry(-1.4566450768814185) q[1];
cx q[0],q[1];
ry(-0.17561118461551484) q[0];
ry(0.5285189974455505) q[1];
cx q[0],q[1];
ry(-2.5550320380720577) q[0];
ry(-0.6208012564705045) q[2];
cx q[0],q[2];
ry(2.5649286795481365) q[0];
ry(-0.8260231830308551) q[2];
cx q[0],q[2];
ry(1.4741556259505157) q[0];
ry(-1.2712523818264314) q[3];
cx q[0],q[3];
ry(-0.3926348033254989) q[0];
ry(-0.7379114170247585) q[3];
cx q[0],q[3];
ry(-2.824511052933159) q[0];
ry(2.225263247840786) q[4];
cx q[0],q[4];
ry(-2.6251484037134025) q[0];
ry(0.7053591760221742) q[4];
cx q[0],q[4];
ry(3.048433472183754) q[0];
ry(-1.6219371813262988) q[5];
cx q[0],q[5];
ry(3.083731024683041) q[0];
ry(2.4444260878964705) q[5];
cx q[0],q[5];
ry(-1.9363862712906226) q[0];
ry(1.089690947814223) q[6];
cx q[0],q[6];
ry(0.26539382592714644) q[0];
ry(1.4575434013178687) q[6];
cx q[0],q[6];
ry(3.113205969208475) q[0];
ry(0.7816135299933021) q[7];
cx q[0],q[7];
ry(1.0547499107485978) q[0];
ry(-0.08254207435683218) q[7];
cx q[0],q[7];
ry(-2.208399497668071) q[1];
ry(-0.7444978032543812) q[2];
cx q[1],q[2];
ry(-3.0982192434439404) q[1];
ry(-1.0179489976013814) q[2];
cx q[1],q[2];
ry(3.1377702735726154) q[1];
ry(-1.725682612499529) q[3];
cx q[1],q[3];
ry(-1.681927421106792) q[1];
ry(-3.0628782607161993) q[3];
cx q[1],q[3];
ry(-1.5538875499659046) q[1];
ry(1.5026253726647338) q[4];
cx q[1],q[4];
ry(-1.9504772959693344) q[1];
ry(2.077967020733539) q[4];
cx q[1],q[4];
ry(2.269718018511832) q[1];
ry(-0.8606906732835367) q[5];
cx q[1],q[5];
ry(1.081297005403189) q[1];
ry(-2.906501094271589) q[5];
cx q[1],q[5];
ry(-2.0029842643489086) q[1];
ry(0.8967024104554591) q[6];
cx q[1],q[6];
ry(0.736805586019468) q[1];
ry(2.773877856485309) q[6];
cx q[1],q[6];
ry(-2.909517535963067) q[1];
ry(1.7056956392747415) q[7];
cx q[1],q[7];
ry(-0.4178248149826027) q[1];
ry(1.9366730079750978) q[7];
cx q[1],q[7];
ry(2.9761722844823066) q[2];
ry(1.9348000725455048) q[3];
cx q[2],q[3];
ry(-2.000778815316367) q[2];
ry(-3.1024180881283603) q[3];
cx q[2],q[3];
ry(1.2606500513037537) q[2];
ry(2.477982019620223) q[4];
cx q[2],q[4];
ry(0.4792684492175674) q[2];
ry(-2.7090227116068664) q[4];
cx q[2],q[4];
ry(2.0668995475275156) q[2];
ry(-0.21152446122635826) q[5];
cx q[2],q[5];
ry(-0.8655160287636087) q[2];
ry(0.2896270081017862) q[5];
cx q[2],q[5];
ry(-1.467387712954186) q[2];
ry(-2.2224132371497265) q[6];
cx q[2],q[6];
ry(-1.3809221251595796) q[2];
ry(2.0858231702610914) q[6];
cx q[2],q[6];
ry(1.5392740963032856) q[2];
ry(-1.452380317988995) q[7];
cx q[2],q[7];
ry(-0.46294617713276653) q[2];
ry(-2.671747969571956) q[7];
cx q[2],q[7];
ry(1.1170199660238012) q[3];
ry(-2.620169562721718) q[4];
cx q[3],q[4];
ry(2.6248927095607995) q[3];
ry(1.6292759105775447) q[4];
cx q[3],q[4];
ry(2.5441664444152754) q[3];
ry(1.5151710130139002) q[5];
cx q[3],q[5];
ry(1.6518079468972084) q[3];
ry(-1.311548741968204) q[5];
cx q[3],q[5];
ry(2.6880315405147432) q[3];
ry(-1.9353717516401632) q[6];
cx q[3],q[6];
ry(-2.6876241313590006) q[3];
ry(2.0064595750931833) q[6];
cx q[3],q[6];
ry(-0.28010514840942763) q[3];
ry(1.5884617224976794) q[7];
cx q[3],q[7];
ry(-1.1094982346638314) q[3];
ry(-0.4965068390617198) q[7];
cx q[3],q[7];
ry(-0.9238929475980973) q[4];
ry(2.8131645792500435) q[5];
cx q[4],q[5];
ry(-1.450067826837592) q[4];
ry(-2.607992571694741) q[5];
cx q[4],q[5];
ry(2.649978795555814) q[4];
ry(1.3895090973647293) q[6];
cx q[4],q[6];
ry(0.6475745696051157) q[4];
ry(1.9924029457720507) q[6];
cx q[4],q[6];
ry(-1.1245849312295697) q[4];
ry(0.9637421775371949) q[7];
cx q[4],q[7];
ry(-1.577927959350225) q[4];
ry(-1.7222620028726352) q[7];
cx q[4],q[7];
ry(2.736938047644242) q[5];
ry(0.005168971343090628) q[6];
cx q[5],q[6];
ry(2.2735953733734515) q[5];
ry(0.3379306324724945) q[6];
cx q[5],q[6];
ry(-2.7301790369617724) q[5];
ry(0.6995011956982814) q[7];
cx q[5],q[7];
ry(-2.890270669833191) q[5];
ry(-0.9087653466275394) q[7];
cx q[5],q[7];
ry(0.6962990084543734) q[6];
ry(1.3775530703505314) q[7];
cx q[6],q[7];
ry(0.21134260673630892) q[6];
ry(-0.9107963193289512) q[7];
cx q[6],q[7];
ry(0.2584840852742304) q[0];
ry(-0.8279734453594978) q[1];
cx q[0],q[1];
ry(1.9446081439272005) q[0];
ry(-2.9701792474114423) q[1];
cx q[0],q[1];
ry(-3.0681650173870842) q[0];
ry(-1.4630030328377508) q[2];
cx q[0],q[2];
ry(0.28449373064369393) q[0];
ry(-2.797232209238059) q[2];
cx q[0],q[2];
ry(1.8573621130733837) q[0];
ry(-2.4524024634074975) q[3];
cx q[0],q[3];
ry(0.7723756529246467) q[0];
ry(-2.4376311134899447) q[3];
cx q[0],q[3];
ry(2.958553197804937) q[0];
ry(-0.09964503413945902) q[4];
cx q[0],q[4];
ry(-0.4358805105039876) q[0];
ry(1.5044518741185697) q[4];
cx q[0],q[4];
ry(-1.515036789361726) q[0];
ry(-1.9147233067912464) q[5];
cx q[0],q[5];
ry(0.5377228784475259) q[0];
ry(-1.736011580159385) q[5];
cx q[0],q[5];
ry(0.7616199027030223) q[0];
ry(0.8231364799242312) q[6];
cx q[0],q[6];
ry(-1.3260741787463752) q[0];
ry(1.4219491531619415) q[6];
cx q[0],q[6];
ry(-0.28083426380577414) q[0];
ry(2.011484196172865) q[7];
cx q[0],q[7];
ry(-3.125926631609656) q[0];
ry(2.5765906902949403) q[7];
cx q[0],q[7];
ry(0.13975956932271139) q[1];
ry(3.0460511813104474) q[2];
cx q[1],q[2];
ry(2.501190390278375) q[1];
ry(2.031058767251915) q[2];
cx q[1],q[2];
ry(-0.6093189210629447) q[1];
ry(1.930130213130612) q[3];
cx q[1],q[3];
ry(-2.9355254759664424) q[1];
ry(-0.7529702126820708) q[3];
cx q[1],q[3];
ry(-1.4263229247864495) q[1];
ry(-2.5183307013668474) q[4];
cx q[1],q[4];
ry(1.9553068788091235) q[1];
ry(-0.9542647330024536) q[4];
cx q[1],q[4];
ry(1.026750790455453) q[1];
ry(-0.5908198498894089) q[5];
cx q[1],q[5];
ry(-1.8805503362861566) q[1];
ry(1.8044096182575222) q[5];
cx q[1],q[5];
ry(1.507844370255748) q[1];
ry(-1.9530860688967302) q[6];
cx q[1],q[6];
ry(-2.5267576565071805) q[1];
ry(-0.08425190410571114) q[6];
cx q[1],q[6];
ry(-1.1147028632918683) q[1];
ry(2.3023681819880952) q[7];
cx q[1],q[7];
ry(-2.527624366455728) q[1];
ry(1.6693029453908714) q[7];
cx q[1],q[7];
ry(0.7560486174760426) q[2];
ry(1.5993851235588286) q[3];
cx q[2],q[3];
ry(-1.743232214876696) q[2];
ry(-1.8996752163482835) q[3];
cx q[2],q[3];
ry(2.4541762996259644) q[2];
ry(1.9634242138769578) q[4];
cx q[2],q[4];
ry(-0.6840085683756119) q[2];
ry(-0.6223792000057464) q[4];
cx q[2],q[4];
ry(-0.26645354424643136) q[2];
ry(-2.940753437351338) q[5];
cx q[2],q[5];
ry(-0.48392967798711495) q[2];
ry(0.3000626388198411) q[5];
cx q[2],q[5];
ry(2.7889996050414423) q[2];
ry(2.9539714915401447) q[6];
cx q[2],q[6];
ry(0.32340880107143644) q[2];
ry(-1.4842520994537065) q[6];
cx q[2],q[6];
ry(-1.2466707816703002) q[2];
ry(0.6999192599956574) q[7];
cx q[2],q[7];
ry(-2.8230177690407854) q[2];
ry(-2.8367768982169936) q[7];
cx q[2],q[7];
ry(-2.4819733911285815) q[3];
ry(-2.9592873903448105) q[4];
cx q[3],q[4];
ry(-2.046518390890424) q[3];
ry(-0.5010844555966157) q[4];
cx q[3],q[4];
ry(-2.4379382472446145) q[3];
ry(1.9434276124407854) q[5];
cx q[3],q[5];
ry(0.7443579104121237) q[3];
ry(-0.21663079101145458) q[5];
cx q[3],q[5];
ry(-0.0045433103573342855) q[3];
ry(1.237194433527745) q[6];
cx q[3],q[6];
ry(2.2248767850864386) q[3];
ry(-0.5625534024107484) q[6];
cx q[3],q[6];
ry(2.0843587729725743) q[3];
ry(-0.7007229935130962) q[7];
cx q[3],q[7];
ry(-2.1129476766653794) q[3];
ry(-0.008850878911758286) q[7];
cx q[3],q[7];
ry(-1.3742900524383748) q[4];
ry(-0.46601825490429066) q[5];
cx q[4],q[5];
ry(-1.139094896519927) q[4];
ry(2.5013454528230317) q[5];
cx q[4],q[5];
ry(-1.0242721897219873) q[4];
ry(-0.2630065203899985) q[6];
cx q[4],q[6];
ry(-2.4204003773685736) q[4];
ry(1.4388447908792905) q[6];
cx q[4],q[6];
ry(0.9807146015988905) q[4];
ry(1.5519722878774544) q[7];
cx q[4],q[7];
ry(-2.447872715219075) q[4];
ry(-2.0768309067807724) q[7];
cx q[4],q[7];
ry(0.8690512845253776) q[5];
ry(2.840527066840467) q[6];
cx q[5],q[6];
ry(1.1981182305029014) q[5];
ry(1.2309092286709222) q[6];
cx q[5],q[6];
ry(-1.4703923513439874) q[5];
ry(-2.607445716597082) q[7];
cx q[5],q[7];
ry(1.2940519161101776) q[5];
ry(-1.4432514378713792) q[7];
cx q[5],q[7];
ry(-1.331231840460893) q[6];
ry(0.40381566096887145) q[7];
cx q[6],q[7];
ry(-0.73604998789839) q[6];
ry(-1.6775037453768868) q[7];
cx q[6],q[7];
ry(-1.3724145688270109) q[0];
ry(1.3910470829398356) q[1];
cx q[0],q[1];
ry(2.089027419471634) q[0];
ry(-1.159917267088078) q[1];
cx q[0],q[1];
ry(1.0867040597448296) q[0];
ry(-2.763713356464971) q[2];
cx q[0],q[2];
ry(-0.2771412563740563) q[0];
ry(1.6022645820469315) q[2];
cx q[0],q[2];
ry(0.011172627363527532) q[0];
ry(0.28575793396829924) q[3];
cx q[0],q[3];
ry(3.033112545090234) q[0];
ry(0.44862390762983617) q[3];
cx q[0],q[3];
ry(0.16265483286643717) q[0];
ry(1.613968690788373) q[4];
cx q[0],q[4];
ry(0.3314284298382005) q[0];
ry(-3.099439235672187) q[4];
cx q[0],q[4];
ry(-0.314558336077277) q[0];
ry(-1.1290422595209755) q[5];
cx q[0],q[5];
ry(2.8363890152425557) q[0];
ry(2.820136360346614) q[5];
cx q[0],q[5];
ry(3.096552553145766) q[0];
ry(-0.04126873430219536) q[6];
cx q[0],q[6];
ry(1.673640481083435) q[0];
ry(-1.3299534206499048) q[6];
cx q[0],q[6];
ry(-0.530656637622406) q[0];
ry(-1.7097681082900438) q[7];
cx q[0],q[7];
ry(-0.08590783337070748) q[0];
ry(0.24270331924722563) q[7];
cx q[0],q[7];
ry(2.4582938913805816) q[1];
ry(-2.882054054448778) q[2];
cx q[1],q[2];
ry(3.0701427806555937) q[1];
ry(-1.7201106741841607) q[2];
cx q[1],q[2];
ry(0.7491049283968367) q[1];
ry(-0.8506275297836172) q[3];
cx q[1],q[3];
ry(0.17623491169901317) q[1];
ry(-1.658641544200199) q[3];
cx q[1],q[3];
ry(1.5810697081979281) q[1];
ry(0.3576705750167095) q[4];
cx q[1],q[4];
ry(-0.1099650004885483) q[1];
ry(-0.7944049657255787) q[4];
cx q[1],q[4];
ry(2.4760871607688166) q[1];
ry(-2.669826604558218) q[5];
cx q[1],q[5];
ry(1.2115289695496816) q[1];
ry(-1.5819165212000712) q[5];
cx q[1],q[5];
ry(-1.3663559577937505) q[1];
ry(0.24614029084671257) q[6];
cx q[1],q[6];
ry(-1.591082609367806) q[1];
ry(1.4191036250116191) q[6];
cx q[1],q[6];
ry(0.281234202982124) q[1];
ry(2.599800750703128) q[7];
cx q[1],q[7];
ry(1.6128759245614237) q[1];
ry(-2.744072149445024) q[7];
cx q[1],q[7];
ry(-1.0262682070480258) q[2];
ry(0.7113739107148483) q[3];
cx q[2],q[3];
ry(1.4594348054739317) q[2];
ry(-2.742966498808331) q[3];
cx q[2],q[3];
ry(2.8970238078909145) q[2];
ry(0.9730958063015686) q[4];
cx q[2],q[4];
ry(2.764820768686529) q[2];
ry(2.1342949322473785) q[4];
cx q[2],q[4];
ry(2.6799309329993837) q[2];
ry(0.7562130552030908) q[5];
cx q[2],q[5];
ry(-0.18968907824692405) q[2];
ry(1.6758554425812295) q[5];
cx q[2],q[5];
ry(-2.3104780058416985) q[2];
ry(-1.7691768960721281) q[6];
cx q[2],q[6];
ry(-1.9271699594346092) q[2];
ry(-2.8400795666340515) q[6];
cx q[2],q[6];
ry(-0.236033772079396) q[2];
ry(0.04531536269701775) q[7];
cx q[2],q[7];
ry(0.621847571388475) q[2];
ry(1.0980866876906452) q[7];
cx q[2],q[7];
ry(1.5375791833069912) q[3];
ry(-1.314440535376752) q[4];
cx q[3],q[4];
ry(0.3878023207224919) q[3];
ry(2.720111166554235) q[4];
cx q[3],q[4];
ry(-1.366225023253313) q[3];
ry(1.2434321232580714) q[5];
cx q[3],q[5];
ry(-2.9341435598954417) q[3];
ry(-0.8059387797603477) q[5];
cx q[3],q[5];
ry(1.30558207092047) q[3];
ry(-2.3752830273224674) q[6];
cx q[3],q[6];
ry(2.687823140096525) q[3];
ry(-2.522804662937894) q[6];
cx q[3],q[6];
ry(0.576649906090611) q[3];
ry(1.1146496617956367) q[7];
cx q[3],q[7];
ry(-0.07408484969754166) q[3];
ry(0.6981704460700018) q[7];
cx q[3],q[7];
ry(0.08044342315325093) q[4];
ry(-2.887418739262635) q[5];
cx q[4],q[5];
ry(-2.669310911935176) q[4];
ry(2.991467068652428) q[5];
cx q[4],q[5];
ry(2.8798565152371736) q[4];
ry(0.9663688772645189) q[6];
cx q[4],q[6];
ry(-2.757350022068049) q[4];
ry(2.5998100238645887) q[6];
cx q[4],q[6];
ry(1.0512526850496027) q[4];
ry(-3.0745384599592294) q[7];
cx q[4],q[7];
ry(2.085805710201867) q[4];
ry(2.0009304067668516) q[7];
cx q[4],q[7];
ry(2.5044190708993708) q[5];
ry(-0.08700700440701352) q[6];
cx q[5],q[6];
ry(0.8495157470997579) q[5];
ry(2.3236927454905216) q[6];
cx q[5],q[6];
ry(-2.6156925387999985) q[5];
ry(-2.4729832173352424) q[7];
cx q[5],q[7];
ry(1.9391121203634534) q[5];
ry(1.285838047557128) q[7];
cx q[5],q[7];
ry(-2.3223020906097545) q[6];
ry(-2.206798515287069) q[7];
cx q[6],q[7];
ry(2.591357848869661) q[6];
ry(1.990822776109002) q[7];
cx q[6],q[7];
ry(-2.1939350501573065) q[0];
ry(0.3132052320755811) q[1];
cx q[0],q[1];
ry(-0.6299336212360513) q[0];
ry(-3.050132979222031) q[1];
cx q[0],q[1];
ry(-1.312292887385081) q[0];
ry(-1.8231772524226346) q[2];
cx q[0],q[2];
ry(-2.9757066258423235) q[0];
ry(2.096048839601196) q[2];
cx q[0],q[2];
ry(0.3623665804022185) q[0];
ry(-2.407321919918045) q[3];
cx q[0],q[3];
ry(-1.3524998524070615) q[0];
ry(0.948029383334852) q[3];
cx q[0],q[3];
ry(0.4996592860458773) q[0];
ry(2.075923963772549) q[4];
cx q[0],q[4];
ry(-2.8126271571110544) q[0];
ry(2.3166017791410978) q[4];
cx q[0],q[4];
ry(-1.7658946268632836) q[0];
ry(0.34707971549973227) q[5];
cx q[0],q[5];
ry(0.9858563309463592) q[0];
ry(-0.19350999395463653) q[5];
cx q[0],q[5];
ry(0.9767391944128612) q[0];
ry(-0.029204738699778684) q[6];
cx q[0],q[6];
ry(-2.601177762894186) q[0];
ry(1.0902625010633058) q[6];
cx q[0],q[6];
ry(-1.2277608512024742) q[0];
ry(-0.5870207661060536) q[7];
cx q[0],q[7];
ry(0.1233300030426463) q[0];
ry(1.6534101153916057) q[7];
cx q[0],q[7];
ry(0.682533821651319) q[1];
ry(2.9212379866832925) q[2];
cx q[1],q[2];
ry(2.559536221070512) q[1];
ry(0.8401179856737191) q[2];
cx q[1],q[2];
ry(2.5754778126196887) q[1];
ry(-1.8981766419079604) q[3];
cx q[1],q[3];
ry(3.07523828902847) q[1];
ry(-2.748737274069332) q[3];
cx q[1],q[3];
ry(-1.1478834682770052) q[1];
ry(-1.6612212912250053) q[4];
cx q[1],q[4];
ry(1.6937609807211995) q[1];
ry(-2.7481687145911438) q[4];
cx q[1],q[4];
ry(3.073571586542642) q[1];
ry(0.2645303153720686) q[5];
cx q[1],q[5];
ry(2.9465119069883627) q[1];
ry(1.4126793103487616) q[5];
cx q[1],q[5];
ry(1.61465562316639) q[1];
ry(-2.0964474556706874) q[6];
cx q[1],q[6];
ry(2.7735540781623764) q[1];
ry(-1.9390416958194234) q[6];
cx q[1],q[6];
ry(-0.7434130048897254) q[1];
ry(0.5433711570900677) q[7];
cx q[1],q[7];
ry(2.0045335898656553) q[1];
ry(0.9241087495352146) q[7];
cx q[1],q[7];
ry(2.581524672084187) q[2];
ry(1.593063663665534) q[3];
cx q[2],q[3];
ry(-1.6290577433560807) q[2];
ry(1.2074035443245732) q[3];
cx q[2],q[3];
ry(1.9203938076450946) q[2];
ry(0.9230026146706117) q[4];
cx q[2],q[4];
ry(0.18035041869624177) q[2];
ry(1.156096449181108) q[4];
cx q[2],q[4];
ry(2.620041329533896) q[2];
ry(-0.1595588386865956) q[5];
cx q[2],q[5];
ry(0.2752946073122643) q[2];
ry(0.17307252026257042) q[5];
cx q[2],q[5];
ry(-1.019598880305911) q[2];
ry(-2.910995393363524) q[6];
cx q[2],q[6];
ry(1.606793401925187) q[2];
ry(0.05266720660934761) q[6];
cx q[2],q[6];
ry(1.845628086833701) q[2];
ry(-2.9349021346381092) q[7];
cx q[2],q[7];
ry(-2.365047560256241) q[2];
ry(-1.412158473319896) q[7];
cx q[2],q[7];
ry(-1.622909279580033) q[3];
ry(-1.389284095593042) q[4];
cx q[3],q[4];
ry(0.8649551260644842) q[3];
ry(2.987153655767852) q[4];
cx q[3],q[4];
ry(-1.677875295704096) q[3];
ry(-2.0745512292299964) q[5];
cx q[3],q[5];
ry(-2.114464164473219) q[3];
ry(-1.6870071721193574) q[5];
cx q[3],q[5];
ry(0.9497456626608679) q[3];
ry(-2.0000271332533046) q[6];
cx q[3],q[6];
ry(-2.3264721753057964) q[3];
ry(0.23192520060076838) q[6];
cx q[3],q[6];
ry(2.658305839726646) q[3];
ry(2.6296852048290544) q[7];
cx q[3],q[7];
ry(-1.0015415946538644) q[3];
ry(-1.2812099908246877) q[7];
cx q[3],q[7];
ry(0.10728965975706828) q[4];
ry(2.6456827515539114) q[5];
cx q[4],q[5];
ry(1.136931913553271) q[4];
ry(0.05178067154972424) q[5];
cx q[4],q[5];
ry(0.9244244883384223) q[4];
ry(-1.1180138490864664) q[6];
cx q[4],q[6];
ry(2.859913356761367) q[4];
ry(-2.0560232639184717) q[6];
cx q[4],q[6];
ry(-0.34094299094125763) q[4];
ry(2.715220800338848) q[7];
cx q[4],q[7];
ry(-2.295707962800752) q[4];
ry(0.9877506666417799) q[7];
cx q[4],q[7];
ry(-1.0679162458561664) q[5];
ry(-2.2375899418282286) q[6];
cx q[5],q[6];
ry(-2.1672163124697494) q[5];
ry(2.814935012772238) q[6];
cx q[5],q[6];
ry(2.443687893067319) q[5];
ry(0.4375732138084418) q[7];
cx q[5],q[7];
ry(-1.3622825540512409) q[5];
ry(1.814272982361043) q[7];
cx q[5],q[7];
ry(-1.2684199592549685) q[6];
ry(2.581530291612598) q[7];
cx q[6],q[7];
ry(2.587041213529253) q[6];
ry(1.4455763442409095) q[7];
cx q[6],q[7];
ry(2.188895455520207) q[0];
ry(0.222466540259588) q[1];
cx q[0],q[1];
ry(-0.4163022858414143) q[0];
ry(-1.3816850565474474) q[1];
cx q[0],q[1];
ry(-0.13793192184883085) q[0];
ry(1.3266996335043308) q[2];
cx q[0],q[2];
ry(-1.0442410607209327) q[0];
ry(1.9952996407111687) q[2];
cx q[0],q[2];
ry(1.363912282174783) q[0];
ry(-1.9127219375193993) q[3];
cx q[0],q[3];
ry(-2.7662412453687937) q[0];
ry(0.9286218635509026) q[3];
cx q[0],q[3];
ry(2.5599568577421445) q[0];
ry(-2.7101739840676684) q[4];
cx q[0],q[4];
ry(0.6577811469619557) q[0];
ry(0.9271227669093571) q[4];
cx q[0],q[4];
ry(2.9831783653679307) q[0];
ry(-0.6434358821485768) q[5];
cx q[0],q[5];
ry(0.9711021461256779) q[0];
ry(0.3785574107330518) q[5];
cx q[0],q[5];
ry(-2.0709992846004237) q[0];
ry(-2.4583277560014385) q[6];
cx q[0],q[6];
ry(0.5634788722847018) q[0];
ry(0.21558615379288648) q[6];
cx q[0],q[6];
ry(-0.17811896684832876) q[0];
ry(2.674511045465446) q[7];
cx q[0],q[7];
ry(2.562155280960235) q[0];
ry(-1.6655556285852884) q[7];
cx q[0],q[7];
ry(2.965192185863276) q[1];
ry(0.6274840412771351) q[2];
cx q[1],q[2];
ry(-1.6989558426107356) q[1];
ry(1.8412531323509196) q[2];
cx q[1],q[2];
ry(1.7127677466455824) q[1];
ry(1.3577377789400789) q[3];
cx q[1],q[3];
ry(0.7257803954220838) q[1];
ry(3.0541240328797636) q[3];
cx q[1],q[3];
ry(1.2696838034971802) q[1];
ry(0.263400635680652) q[4];
cx q[1],q[4];
ry(-1.9149382946093008) q[1];
ry(0.39914365452425393) q[4];
cx q[1],q[4];
ry(-0.9813372197664751) q[1];
ry(-2.315006671162935) q[5];
cx q[1],q[5];
ry(-1.1730098952399848) q[1];
ry(-2.848534259094) q[5];
cx q[1],q[5];
ry(-1.3250218443816868) q[1];
ry(-1.6246994417785305) q[6];
cx q[1],q[6];
ry(2.4109287694263384) q[1];
ry(-0.6753510655797079) q[6];
cx q[1],q[6];
ry(-2.135599847590289) q[1];
ry(0.5739090665152276) q[7];
cx q[1],q[7];
ry(0.7351099063636067) q[1];
ry(-0.9632090834812541) q[7];
cx q[1],q[7];
ry(0.4494684287203947) q[2];
ry(-1.8167704587411695) q[3];
cx q[2],q[3];
ry(-1.293373560459548) q[2];
ry(-0.957579534344398) q[3];
cx q[2],q[3];
ry(3.123518601540859) q[2];
ry(-3.0934712408202034) q[4];
cx q[2],q[4];
ry(0.7539819678651106) q[2];
ry(1.9505438536646245) q[4];
cx q[2],q[4];
ry(1.0018284775883712) q[2];
ry(-2.3595934017106295) q[5];
cx q[2],q[5];
ry(0.2687805864674198) q[2];
ry(-1.632717357131767) q[5];
cx q[2],q[5];
ry(1.0461043508226735) q[2];
ry(2.0070064679788366) q[6];
cx q[2],q[6];
ry(-2.550044878045905) q[2];
ry(-2.404004540063083) q[6];
cx q[2],q[6];
ry(2.322958534467855) q[2];
ry(-1.6557643226692722) q[7];
cx q[2],q[7];
ry(-3.1106026966519766) q[2];
ry(0.16087190549094643) q[7];
cx q[2],q[7];
ry(2.2740683687136167) q[3];
ry(2.683384808111303) q[4];
cx q[3],q[4];
ry(2.0409396586431194) q[3];
ry(2.0026045829375754) q[4];
cx q[3],q[4];
ry(-0.18463535149398727) q[3];
ry(-0.9348019048398807) q[5];
cx q[3],q[5];
ry(1.8238169910576438) q[3];
ry(-0.681462287971774) q[5];
cx q[3],q[5];
ry(-3.0583191041927216) q[3];
ry(-1.951599562205157) q[6];
cx q[3],q[6];
ry(-1.3244091249273833) q[3];
ry(2.989296526246567) q[6];
cx q[3],q[6];
ry(1.7545800069843904) q[3];
ry(-2.2841090887871567) q[7];
cx q[3],q[7];
ry(2.0278657185523956) q[3];
ry(-1.0844209705244867) q[7];
cx q[3],q[7];
ry(1.074783167041244) q[4];
ry(-0.14540466762359724) q[5];
cx q[4],q[5];
ry(1.1319951937193407) q[4];
ry(-2.7318092401679888) q[5];
cx q[4],q[5];
ry(-2.694455841477493) q[4];
ry(1.7009746134785262) q[6];
cx q[4],q[6];
ry(1.9969496421655784) q[4];
ry(0.370831396063985) q[6];
cx q[4],q[6];
ry(3.039731906802641) q[4];
ry(2.2393369929473197) q[7];
cx q[4],q[7];
ry(-1.3913019946497152) q[4];
ry(2.464834036505631) q[7];
cx q[4],q[7];
ry(-3.079655817746023) q[5];
ry(-0.6590040762477823) q[6];
cx q[5],q[6];
ry(0.47277842464325964) q[5];
ry(-2.873916013429164) q[6];
cx q[5],q[6];
ry(1.7774897716879154) q[5];
ry(0.9495580824610306) q[7];
cx q[5],q[7];
ry(1.6321667191224822) q[5];
ry(-2.5826130560191363) q[7];
cx q[5],q[7];
ry(-1.563560679341462) q[6];
ry(0.00395060226412438) q[7];
cx q[6],q[7];
ry(-0.9851066034643867) q[6];
ry(2.4051415636821623) q[7];
cx q[6],q[7];
ry(-1.553793358866848) q[0];
ry(2.7231847529750137) q[1];
ry(2.308485112054057) q[2];
ry(1.9089852141766142) q[3];
ry(-2.0497427771554504) q[4];
ry(2.128002822369968) q[5];
ry(2.205090807219354) q[6];
ry(1.3944239768750188) q[7];