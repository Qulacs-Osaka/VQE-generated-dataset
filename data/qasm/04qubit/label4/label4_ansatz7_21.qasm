OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.523105393649561) q[0];
ry(0.6806319967278958) q[1];
cx q[0],q[1];
ry(-1.9108201382204726) q[0];
ry(-0.6285896820674087) q[1];
cx q[0],q[1];
ry(-0.5188680762652114) q[0];
ry(2.399250009955553) q[2];
cx q[0],q[2];
ry(-2.9065138537947295) q[0];
ry(1.6338835116642751) q[2];
cx q[0],q[2];
ry(-2.1511571516919803) q[0];
ry(2.924536556944314) q[3];
cx q[0],q[3];
ry(-3.0246785361640454) q[0];
ry(-2.9088278720616842) q[3];
cx q[0],q[3];
ry(0.6477139853966882) q[1];
ry(-2.337484971260146) q[2];
cx q[1],q[2];
ry(1.459796675344526) q[1];
ry(1.0418601898104063) q[2];
cx q[1],q[2];
ry(0.3682450264669575) q[1];
ry(2.403244099567703) q[3];
cx q[1],q[3];
ry(-2.624994186187834) q[1];
ry(-0.19754493534268036) q[3];
cx q[1],q[3];
ry(-1.8069294527527167) q[2];
ry(1.4252217992609113) q[3];
cx q[2],q[3];
ry(-0.9610553850744106) q[2];
ry(1.1530544012668367) q[3];
cx q[2],q[3];
ry(0.25875300147785557) q[0];
ry(-0.8405092570321893) q[1];
cx q[0],q[1];
ry(-1.5633287749040257) q[0];
ry(2.4268255422628666) q[1];
cx q[0],q[1];
ry(2.256933930449305) q[0];
ry(-3.1347887145027724) q[2];
cx q[0],q[2];
ry(0.5785498134877398) q[0];
ry(1.4751559215123704) q[2];
cx q[0],q[2];
ry(1.0919383282303854) q[0];
ry(-1.461759803846401) q[3];
cx q[0],q[3];
ry(2.1442992145888633) q[0];
ry(2.3173095336419713) q[3];
cx q[0],q[3];
ry(-0.17167052778512634) q[1];
ry(-2.6420962276013227) q[2];
cx q[1],q[2];
ry(-2.4107834785861364) q[1];
ry(-2.3604813221839738) q[2];
cx q[1],q[2];
ry(-1.6806536883105743) q[1];
ry(0.5634263035007789) q[3];
cx q[1],q[3];
ry(1.1436723763572318) q[1];
ry(0.47465872133464515) q[3];
cx q[1],q[3];
ry(1.5781347131158174) q[2];
ry(-2.9572689026789973) q[3];
cx q[2],q[3];
ry(-1.0359854174402026) q[2];
ry(1.5437688124295337) q[3];
cx q[2],q[3];
ry(0.7839093096149723) q[0];
ry(-0.36017215279009906) q[1];
cx q[0],q[1];
ry(1.2696789985624921) q[0];
ry(-1.4335550634085241) q[1];
cx q[0],q[1];
ry(-2.412557082419557) q[0];
ry(-2.8210034151794936) q[2];
cx q[0],q[2];
ry(-2.021250821842548) q[0];
ry(0.5552125166207847) q[2];
cx q[0],q[2];
ry(-1.4800163599328713) q[0];
ry(-0.25225629336430966) q[3];
cx q[0],q[3];
ry(-0.09450899926400388) q[0];
ry(0.5560637942166942) q[3];
cx q[0],q[3];
ry(0.5460195146818023) q[1];
ry(-1.7828727978964887) q[2];
cx q[1],q[2];
ry(1.9473496145387559) q[1];
ry(-2.37967164536106) q[2];
cx q[1],q[2];
ry(1.9291554107131081) q[1];
ry(0.32042101909390747) q[3];
cx q[1],q[3];
ry(2.136383438370604) q[1];
ry(0.12395809420358535) q[3];
cx q[1],q[3];
ry(3.02375960905545) q[2];
ry(1.410923346055382) q[3];
cx q[2],q[3];
ry(-1.8459290470974414) q[2];
ry(1.2011404598897828) q[3];
cx q[2],q[3];
ry(-1.7990717118583888) q[0];
ry(-1.6389529156447429) q[1];
cx q[0],q[1];
ry(2.2872739902380927) q[0];
ry(-2.0236904879875013) q[1];
cx q[0],q[1];
ry(-1.1475703385997118) q[0];
ry(1.3538950064907638) q[2];
cx q[0],q[2];
ry(-1.6026435331657873) q[0];
ry(2.5379714113574527) q[2];
cx q[0],q[2];
ry(2.0495150267437747) q[0];
ry(-1.4423579863198888) q[3];
cx q[0],q[3];
ry(-2.9064946505193103) q[0];
ry(-2.8549476313765116) q[3];
cx q[0],q[3];
ry(2.128844396444646) q[1];
ry(-0.48136575053406006) q[2];
cx q[1],q[2];
ry(-0.7991114959969365) q[1];
ry(2.892227109747055) q[2];
cx q[1],q[2];
ry(-0.19642265965213745) q[1];
ry(-0.0226050383562324) q[3];
cx q[1],q[3];
ry(-2.1625216238088694) q[1];
ry(-0.2909496359383432) q[3];
cx q[1],q[3];
ry(-0.21227859402068833) q[2];
ry(2.3938277859352) q[3];
cx q[2],q[3];
ry(2.877305436884612) q[2];
ry(2.1362670469350418) q[3];
cx q[2],q[3];
ry(-0.08106297129590345) q[0];
ry(0.8095838267411113) q[1];
cx q[0],q[1];
ry(1.5419996969516139) q[0];
ry(1.0622303856417121) q[1];
cx q[0],q[1];
ry(-2.8288608095600196) q[0];
ry(-0.6138999587456805) q[2];
cx q[0],q[2];
ry(-0.036293304432657804) q[0];
ry(1.0568327217154918) q[2];
cx q[0],q[2];
ry(-1.7373274717601794) q[0];
ry(-0.4049376003857021) q[3];
cx q[0],q[3];
ry(1.9805914414504584) q[0];
ry(-3.1201084228297047) q[3];
cx q[0],q[3];
ry(-2.8564178706221672) q[1];
ry(2.724526802397603) q[2];
cx q[1],q[2];
ry(2.2884699853822434) q[1];
ry(-0.7766127198193251) q[2];
cx q[1],q[2];
ry(-0.17419371869969158) q[1];
ry(-3.020203496643853) q[3];
cx q[1],q[3];
ry(2.2404832757413917) q[1];
ry(2.412024445305569) q[3];
cx q[1],q[3];
ry(2.614026629493677) q[2];
ry(-2.2258409197824442) q[3];
cx q[2],q[3];
ry(-2.192741404486439) q[2];
ry(-0.6249821979208907) q[3];
cx q[2],q[3];
ry(1.9836683884942772) q[0];
ry(0.8324256878719596) q[1];
cx q[0],q[1];
ry(-1.9928141735041516) q[0];
ry(1.255536177143731) q[1];
cx q[0],q[1];
ry(-0.514171089428886) q[0];
ry(0.5217014771772289) q[2];
cx q[0],q[2];
ry(-2.5994670858114337) q[0];
ry(2.2535013618436546) q[2];
cx q[0],q[2];
ry(-2.839631285541617) q[0];
ry(2.5770970178552823) q[3];
cx q[0],q[3];
ry(-1.71489052425699) q[0];
ry(-3.0983220949578776) q[3];
cx q[0],q[3];
ry(-0.09776673079770021) q[1];
ry(1.0081525909953752) q[2];
cx q[1],q[2];
ry(-1.4516540591496625) q[1];
ry(2.282312636115477) q[2];
cx q[1],q[2];
ry(-1.318074331333091) q[1];
ry(1.5248201834937731) q[3];
cx q[1],q[3];
ry(2.625571934625845) q[1];
ry(-1.9405596131632825) q[3];
cx q[1],q[3];
ry(0.5700676449463957) q[2];
ry(1.1289299103890533) q[3];
cx q[2],q[3];
ry(0.47880603781321107) q[2];
ry(-0.8611416490474655) q[3];
cx q[2],q[3];
ry(-1.3391237669684033) q[0];
ry(2.9534594798047156) q[1];
cx q[0],q[1];
ry(-0.6296740457835606) q[0];
ry(-0.654370994387226) q[1];
cx q[0],q[1];
ry(-0.6321744947502035) q[0];
ry(-0.9869630395702993) q[2];
cx q[0],q[2];
ry(-2.9542563770432126) q[0];
ry(0.06592121714440227) q[2];
cx q[0],q[2];
ry(-0.5862030588180805) q[0];
ry(-2.756396110700507) q[3];
cx q[0],q[3];
ry(-1.398718011954574) q[0];
ry(-1.508580530132227) q[3];
cx q[0],q[3];
ry(-0.1927953062118073) q[1];
ry(-2.7107006736271138) q[2];
cx q[1],q[2];
ry(0.25730747191648273) q[1];
ry(0.14288537754186215) q[2];
cx q[1],q[2];
ry(2.049399324316914) q[1];
ry(1.781917320930861) q[3];
cx q[1],q[3];
ry(1.9476788977370907) q[1];
ry(-2.160135941075559) q[3];
cx q[1],q[3];
ry(-2.5558823988460047) q[2];
ry(0.47799104781412705) q[3];
cx q[2],q[3];
ry(-2.0656406382509314) q[2];
ry(-2.672139559428375) q[3];
cx q[2],q[3];
ry(0.17356154282346292) q[0];
ry(0.6017499258779365) q[1];
cx q[0],q[1];
ry(-1.9930683556984112) q[0];
ry(0.060499422443907165) q[1];
cx q[0],q[1];
ry(2.5375894628096236) q[0];
ry(1.3138930979863783) q[2];
cx q[0],q[2];
ry(2.951696215621554) q[0];
ry(1.0524462336676859) q[2];
cx q[0],q[2];
ry(1.453607464549718) q[0];
ry(-1.187426486418659) q[3];
cx q[0],q[3];
ry(2.2008550655634753) q[0];
ry(0.17254760418207088) q[3];
cx q[0],q[3];
ry(0.601326773129159) q[1];
ry(0.7444287783200126) q[2];
cx q[1],q[2];
ry(-1.3168545430913579) q[1];
ry(0.625915706608672) q[2];
cx q[1],q[2];
ry(2.442864436968929) q[1];
ry(1.4556915419137848) q[3];
cx q[1],q[3];
ry(0.47435491923913103) q[1];
ry(-2.1860030120065765) q[3];
cx q[1],q[3];
ry(2.1458991266374543) q[2];
ry(1.3674620058895224) q[3];
cx q[2],q[3];
ry(-0.6370495054178091) q[2];
ry(-2.7092545642533) q[3];
cx q[2],q[3];
ry(-2.3281945973735927) q[0];
ry(-1.3290721780930486) q[1];
cx q[0],q[1];
ry(-1.2551621489923879) q[0];
ry(2.912825151639051) q[1];
cx q[0],q[1];
ry(-0.8681801296170675) q[0];
ry(1.732256816581929) q[2];
cx q[0],q[2];
ry(-2.6360160578471477) q[0];
ry(0.9102577605086024) q[2];
cx q[0],q[2];
ry(1.1355221385737577) q[0];
ry(0.5509850249005623) q[3];
cx q[0],q[3];
ry(-3.1361740963256968) q[0];
ry(0.21196275195601808) q[3];
cx q[0],q[3];
ry(1.4419923788534672) q[1];
ry(-0.45333345472034825) q[2];
cx q[1],q[2];
ry(-0.37758178076958343) q[1];
ry(0.5834627834575361) q[2];
cx q[1],q[2];
ry(-2.542712968721353) q[1];
ry(1.4568863510769114) q[3];
cx q[1],q[3];
ry(-1.1208734610671405) q[1];
ry(2.069947179066544) q[3];
cx q[1],q[3];
ry(-1.2161588964340266) q[2];
ry(-0.6594404925966959) q[3];
cx q[2],q[3];
ry(1.3875545824302922) q[2];
ry(2.129480663436878) q[3];
cx q[2],q[3];
ry(2.928088205733787) q[0];
ry(-0.26695983449553456) q[1];
cx q[0],q[1];
ry(-0.6965220543829715) q[0];
ry(-2.6992368204001314) q[1];
cx q[0],q[1];
ry(-2.0172463724290077) q[0];
ry(1.1914554809333766) q[2];
cx q[0],q[2];
ry(-2.874550603678696) q[0];
ry(-0.14111492823237626) q[2];
cx q[0],q[2];
ry(0.3210581181319707) q[0];
ry(2.2948733004675437) q[3];
cx q[0],q[3];
ry(-0.31256704564812215) q[0];
ry(1.547412445921296) q[3];
cx q[0],q[3];
ry(1.2554903881277506) q[1];
ry(-3.0415114153728617) q[2];
cx q[1],q[2];
ry(-0.6321045315965037) q[1];
ry(-1.3163818950188888) q[2];
cx q[1],q[2];
ry(-0.4267633608553571) q[1];
ry(-0.199659049239846) q[3];
cx q[1],q[3];
ry(-2.48859839313192) q[1];
ry(-0.356454148941995) q[3];
cx q[1],q[3];
ry(0.9888844949027554) q[2];
ry(-2.9849829132305126) q[3];
cx q[2],q[3];
ry(1.2458177255418865) q[2];
ry(2.595067979927305) q[3];
cx q[2],q[3];
ry(-2.739112450367798) q[0];
ry(1.0981627881873397) q[1];
cx q[0],q[1];
ry(-0.8382339477404805) q[0];
ry(3.025004594101439) q[1];
cx q[0],q[1];
ry(-2.386914002110349) q[0];
ry(-2.521532242780437) q[2];
cx q[0],q[2];
ry(-2.2229844194815893) q[0];
ry(0.2531962783191445) q[2];
cx q[0],q[2];
ry(0.4164561084266314) q[0];
ry(0.5915266587915484) q[3];
cx q[0],q[3];
ry(2.0625677295274274) q[0];
ry(-3.1074257522491515) q[3];
cx q[0],q[3];
ry(-1.3964398236116917) q[1];
ry(2.436470353823509) q[2];
cx q[1],q[2];
ry(0.9110923715080537) q[1];
ry(2.8632880549335864) q[2];
cx q[1],q[2];
ry(-2.9698668481928894) q[1];
ry(2.446566046709873) q[3];
cx q[1],q[3];
ry(2.437185560941679) q[1];
ry(0.4034828447848601) q[3];
cx q[1],q[3];
ry(1.152893191365488) q[2];
ry(2.8748201213663664) q[3];
cx q[2],q[3];
ry(0.9569034519469373) q[2];
ry(-3.137169380276088) q[3];
cx q[2],q[3];
ry(-2.534143665507871) q[0];
ry(-2.5997525504636743) q[1];
cx q[0],q[1];
ry(-2.142945967816538) q[0];
ry(-1.102635368575797) q[1];
cx q[0],q[1];
ry(2.4590270042579525) q[0];
ry(-1.8357300452109482) q[2];
cx q[0],q[2];
ry(-1.307084867035535) q[0];
ry(1.1990440857505478) q[2];
cx q[0],q[2];
ry(-0.5521277141162924) q[0];
ry(1.2085740797251985) q[3];
cx q[0],q[3];
ry(3.066159196472784) q[0];
ry(-3.036648731904935) q[3];
cx q[0],q[3];
ry(0.5449193668643222) q[1];
ry(0.3923477595945482) q[2];
cx q[1],q[2];
ry(-1.279435721869233) q[1];
ry(-1.2496401750375736) q[2];
cx q[1],q[2];
ry(-2.351652460783921) q[1];
ry(1.619251057332538) q[3];
cx q[1],q[3];
ry(2.9166501759037686) q[1];
ry(-0.6027485788184226) q[3];
cx q[1],q[3];
ry(-2.8577219969101333) q[2];
ry(2.2402429588405566) q[3];
cx q[2],q[3];
ry(-1.466456372680209) q[2];
ry(-0.5360542952441163) q[3];
cx q[2],q[3];
ry(1.8978228201854943) q[0];
ry(-2.843748714014952) q[1];
cx q[0],q[1];
ry(2.161356511002626) q[0];
ry(-1.2692386040498267) q[1];
cx q[0],q[1];
ry(-0.30570837656804617) q[0];
ry(-1.9995936249070367) q[2];
cx q[0],q[2];
ry(-2.0497905884659193) q[0];
ry(-0.5364016311115849) q[2];
cx q[0],q[2];
ry(1.0967108385537507) q[0];
ry(0.9712835622415277) q[3];
cx q[0],q[3];
ry(-1.7190725715788986) q[0];
ry(0.48377166030863533) q[3];
cx q[0],q[3];
ry(-0.9359517124204011) q[1];
ry(0.5750034028746525) q[2];
cx q[1],q[2];
ry(-2.433869161298007) q[1];
ry(-2.1144102805033698) q[2];
cx q[1],q[2];
ry(0.6463261781145846) q[1];
ry(-2.6050325940049794) q[3];
cx q[1],q[3];
ry(1.5091765890764908) q[1];
ry(-2.045992875316374) q[3];
cx q[1],q[3];
ry(-2.764811846544477) q[2];
ry(-2.304733064091429) q[3];
cx q[2],q[3];
ry(2.4452620657350916) q[2];
ry(-0.13821687628860335) q[3];
cx q[2],q[3];
ry(-1.5354838465886707) q[0];
ry(1.6328111920761472) q[1];
cx q[0],q[1];
ry(-2.5296174143342416) q[0];
ry(1.3819771167592123) q[1];
cx q[0],q[1];
ry(0.13917102999329423) q[0];
ry(2.3590024789941046) q[2];
cx q[0],q[2];
ry(-2.6187842200114386) q[0];
ry(1.827452081229011) q[2];
cx q[0],q[2];
ry(-1.4326035453890382) q[0];
ry(-0.3629955920779375) q[3];
cx q[0],q[3];
ry(1.2330887624183928) q[0];
ry(2.6523037981774586) q[3];
cx q[0],q[3];
ry(1.7208616601231725) q[1];
ry(-0.6603745562606473) q[2];
cx q[1],q[2];
ry(-1.0393622926342079) q[1];
ry(0.34212472839056307) q[2];
cx q[1],q[2];
ry(-2.7250346496231703) q[1];
ry(2.7491965704817485) q[3];
cx q[1],q[3];
ry(-1.8214024238398991) q[1];
ry(-1.2091264130761243) q[3];
cx q[1],q[3];
ry(1.5035099624235393) q[2];
ry(-0.028246146650195824) q[3];
cx q[2],q[3];
ry(-2.9948480227025214) q[2];
ry(-2.105336206530547) q[3];
cx q[2],q[3];
ry(1.7371546021371902) q[0];
ry(1.5392794096967455) q[1];
cx q[0],q[1];
ry(1.164608804731586) q[0];
ry(-1.76886232783081) q[1];
cx q[0],q[1];
ry(-1.7956173096293933) q[0];
ry(1.8778271986381077) q[2];
cx q[0],q[2];
ry(0.20654820349422198) q[0];
ry(0.0071607583437911515) q[2];
cx q[0],q[2];
ry(1.1681115904281258) q[0];
ry(1.9911950728101369) q[3];
cx q[0],q[3];
ry(1.85416262637903) q[0];
ry(2.462112649547104) q[3];
cx q[0],q[3];
ry(2.7994812526847888) q[1];
ry(1.3665621376738826) q[2];
cx q[1],q[2];
ry(2.0136841655699005) q[1];
ry(1.2973924970460784) q[2];
cx q[1],q[2];
ry(0.9851522709650903) q[1];
ry(1.1328970366533264) q[3];
cx q[1],q[3];
ry(2.5607373310117616) q[1];
ry(-3.0264815269115197) q[3];
cx q[1],q[3];
ry(2.6755306683838462) q[2];
ry(3.0899400100847383) q[3];
cx q[2],q[3];
ry(1.3894149015192605) q[2];
ry(-2.7313416117720535) q[3];
cx q[2],q[3];
ry(2.2723057534067177) q[0];
ry(-2.328062282378044) q[1];
cx q[0],q[1];
ry(-0.34745678944354186) q[0];
ry(3.0550048098291853) q[1];
cx q[0],q[1];
ry(-2.6078905667957506) q[0];
ry(1.703028893496926) q[2];
cx q[0],q[2];
ry(0.6952115743471685) q[0];
ry(-2.262268476540661) q[2];
cx q[0],q[2];
ry(0.9402278888767343) q[0];
ry(-2.2795921249988718) q[3];
cx q[0],q[3];
ry(1.4963016657327926) q[0];
ry(0.7723268985318973) q[3];
cx q[0],q[3];
ry(0.4831646696076062) q[1];
ry(2.325629925612497) q[2];
cx q[1],q[2];
ry(-1.6121942257783664) q[1];
ry(2.966675895085974) q[2];
cx q[1],q[2];
ry(0.35929813899002216) q[1];
ry(2.426786711701744) q[3];
cx q[1],q[3];
ry(-0.6840278144617615) q[1];
ry(-1.9177531165409354) q[3];
cx q[1],q[3];
ry(-2.1786047298627795) q[2];
ry(0.1279027216239073) q[3];
cx q[2],q[3];
ry(-2.732081551699895) q[2];
ry(-0.6332532711865592) q[3];
cx q[2],q[3];
ry(0.3010937121269526) q[0];
ry(0.2851411461632898) q[1];
cx q[0],q[1];
ry(0.8256774996203866) q[0];
ry(1.2269793221728482) q[1];
cx q[0],q[1];
ry(0.9401332777720927) q[0];
ry(2.4525371214024068) q[2];
cx q[0],q[2];
ry(3.0847215787568505) q[0];
ry(1.1189662200382058) q[2];
cx q[0],q[2];
ry(2.332511614566382) q[0];
ry(1.5620828363712853) q[3];
cx q[0],q[3];
ry(-0.6328104730982664) q[0];
ry(0.9938562054467323) q[3];
cx q[0],q[3];
ry(0.8055892903306967) q[1];
ry(0.4474341387708609) q[2];
cx q[1],q[2];
ry(-1.0744503014307893) q[1];
ry(2.7231640950914984) q[2];
cx q[1],q[2];
ry(-2.7597167943532375) q[1];
ry(-1.9102510904005356) q[3];
cx q[1],q[3];
ry(0.24656858366797585) q[1];
ry(-2.5902032926177325) q[3];
cx q[1],q[3];
ry(0.17673316383857898) q[2];
ry(-2.4762454845143953) q[3];
cx q[2],q[3];
ry(-1.3877384934410357) q[2];
ry(-2.4403195994578692) q[3];
cx q[2],q[3];
ry(-2.048677213969258) q[0];
ry(-1.906023270915596) q[1];
cx q[0],q[1];
ry(2.7743379539372173) q[0];
ry(0.22161044210249603) q[1];
cx q[0],q[1];
ry(1.5530944707797607) q[0];
ry(0.06391969692032905) q[2];
cx q[0],q[2];
ry(0.7497143069640071) q[0];
ry(-2.041561600586258) q[2];
cx q[0],q[2];
ry(2.1272097863742125) q[0];
ry(-1.56597889769112) q[3];
cx q[0],q[3];
ry(1.5710343384289214) q[0];
ry(1.9553892298438111) q[3];
cx q[0],q[3];
ry(-0.6571966280835814) q[1];
ry(1.3330805609986331) q[2];
cx q[1],q[2];
ry(-2.117950916350507) q[1];
ry(-1.7875180429862692) q[2];
cx q[1],q[2];
ry(2.9825082724682432) q[1];
ry(2.1498284635739275) q[3];
cx q[1],q[3];
ry(0.4006241417626052) q[1];
ry(3.043844965616787) q[3];
cx q[1],q[3];
ry(-2.9529150882107897) q[2];
ry(-2.2634648497187264) q[3];
cx q[2],q[3];
ry(2.895267209210902) q[2];
ry(1.593797873273184) q[3];
cx q[2],q[3];
ry(0.318282387620581) q[0];
ry(1.649930410938364) q[1];
cx q[0],q[1];
ry(2.4368262976870456) q[0];
ry(-0.7727253522065523) q[1];
cx q[0],q[1];
ry(-1.913182938208056) q[0];
ry(-0.9834667968333538) q[2];
cx q[0],q[2];
ry(-1.2135176207885676) q[0];
ry(1.4526567178748158) q[2];
cx q[0],q[2];
ry(2.908300745853349) q[0];
ry(-1.0578260978003835) q[3];
cx q[0],q[3];
ry(0.13319463219616612) q[0];
ry(-0.2341732702687452) q[3];
cx q[0],q[3];
ry(2.6625033874604993) q[1];
ry(-2.807956167746126) q[2];
cx q[1],q[2];
ry(-0.018871638922359324) q[1];
ry(-0.29024225553197514) q[2];
cx q[1],q[2];
ry(1.1254024612052058) q[1];
ry(-0.831659629156733) q[3];
cx q[1],q[3];
ry(-2.5993841246275755) q[1];
ry(1.4274333218598292) q[3];
cx q[1],q[3];
ry(1.1257448162437909) q[2];
ry(0.4422433314413796) q[3];
cx q[2],q[3];
ry(-1.7260661953935337) q[2];
ry(1.7207023310831544) q[3];
cx q[2],q[3];
ry(-2.059608828941732) q[0];
ry(-2.4406151929778503) q[1];
cx q[0],q[1];
ry(3.087396447694434) q[0];
ry(2.5347011850922994) q[1];
cx q[0],q[1];
ry(0.0623047521789335) q[0];
ry(2.447522779139469) q[2];
cx q[0],q[2];
ry(1.1136577314692788) q[0];
ry(-1.3016087855362906) q[2];
cx q[0],q[2];
ry(-2.7764890557591246) q[0];
ry(1.1667743155615886) q[3];
cx q[0],q[3];
ry(-0.6573890110279261) q[0];
ry(0.9123706009107018) q[3];
cx q[0],q[3];
ry(-1.278831360235797) q[1];
ry(1.725320129664277) q[2];
cx q[1],q[2];
ry(2.0156220414828625) q[1];
ry(-1.2445829759633191) q[2];
cx q[1],q[2];
ry(0.48647644294247977) q[1];
ry(-2.1428104284562948) q[3];
cx q[1],q[3];
ry(0.6356274944449553) q[1];
ry(-0.19521049133751767) q[3];
cx q[1],q[3];
ry(-1.9043915438676091) q[2];
ry(3.138451633033833) q[3];
cx q[2],q[3];
ry(-2.0374934577185195) q[2];
ry(1.5521451502655184) q[3];
cx q[2],q[3];
ry(0.1619121825757618) q[0];
ry(-0.6991226781677398) q[1];
cx q[0],q[1];
ry(0.5849018669193775) q[0];
ry(-1.9748155574035418) q[1];
cx q[0],q[1];
ry(2.39119017336391) q[0];
ry(2.015095933782418) q[2];
cx q[0],q[2];
ry(0.8129425695093991) q[0];
ry(-0.16605964632203474) q[2];
cx q[0],q[2];
ry(2.1289356923764373) q[0];
ry(1.4309640009077549) q[3];
cx q[0],q[3];
ry(-2.84433390902317) q[0];
ry(-2.700598622696874) q[3];
cx q[0],q[3];
ry(-0.6978691316772402) q[1];
ry(1.4873403349489083) q[2];
cx q[1],q[2];
ry(-1.8161726729433691) q[1];
ry(-1.1768727149445761) q[2];
cx q[1],q[2];
ry(2.466591626911106) q[1];
ry(3.128545563346524) q[3];
cx q[1],q[3];
ry(-1.7518576205228928) q[1];
ry(-0.39233249032307826) q[3];
cx q[1],q[3];
ry(-1.1490957617501447) q[2];
ry(2.1736235499642245) q[3];
cx q[2],q[3];
ry(1.0716312116812254) q[2];
ry(-1.9718193483992232) q[3];
cx q[2],q[3];
ry(0.05406026663403485) q[0];
ry(-2.573753650516064) q[1];
cx q[0],q[1];
ry(-0.39730906339678307) q[0];
ry(-1.4282608842395152) q[1];
cx q[0],q[1];
ry(-0.8534374359202639) q[0];
ry(-0.7470955086155167) q[2];
cx q[0],q[2];
ry(1.1781379395636047) q[0];
ry(1.7154085536593178) q[2];
cx q[0],q[2];
ry(-2.6997118000333575) q[0];
ry(1.2860054322975354) q[3];
cx q[0],q[3];
ry(0.38690182165578424) q[0];
ry(0.09526095830148043) q[3];
cx q[0],q[3];
ry(3.0223972822602443) q[1];
ry(-2.680220573874082) q[2];
cx q[1],q[2];
ry(1.5213024781329387) q[1];
ry(1.8408861470757705) q[2];
cx q[1],q[2];
ry(1.923806866874342) q[1];
ry(1.1419163150022456) q[3];
cx q[1],q[3];
ry(-2.135579564846892) q[1];
ry(-0.7461363029664306) q[3];
cx q[1],q[3];
ry(2.5078139650857545) q[2];
ry(-1.966958793833423) q[3];
cx q[2],q[3];
ry(0.7764144310883198) q[2];
ry(-1.574932661816735) q[3];
cx q[2],q[3];
ry(-1.2124190786796785) q[0];
ry(1.8896943079145612) q[1];
cx q[0],q[1];
ry(2.2638574165392926) q[0];
ry(1.022676326482963) q[1];
cx q[0],q[1];
ry(2.0138935974421153) q[0];
ry(-0.27483102126992964) q[2];
cx q[0],q[2];
ry(2.8887413587025055) q[0];
ry(0.6351403016334451) q[2];
cx q[0],q[2];
ry(-0.1692933020203622) q[0];
ry(0.5234344327421782) q[3];
cx q[0],q[3];
ry(2.5111319706002666) q[0];
ry(2.613436109550824) q[3];
cx q[0],q[3];
ry(-2.8920672386304895) q[1];
ry(-2.7976973742948217) q[2];
cx q[1],q[2];
ry(0.9175412495081685) q[1];
ry(-2.4041072751388666) q[2];
cx q[1],q[2];
ry(0.09247015604244752) q[1];
ry(-1.0198562380820215) q[3];
cx q[1],q[3];
ry(2.7670084799242076) q[1];
ry(0.33552375519472477) q[3];
cx q[1],q[3];
ry(-0.6297471848216178) q[2];
ry(-1.1484464522026743) q[3];
cx q[2],q[3];
ry(0.3698829827471042) q[2];
ry(-3.078355634256705) q[3];
cx q[2],q[3];
ry(2.138686415537587) q[0];
ry(-0.06948311198984314) q[1];
cx q[0],q[1];
ry(2.3032364154759457) q[0];
ry(-0.6551250750199645) q[1];
cx q[0],q[1];
ry(1.2823350162808147) q[0];
ry(-2.7109801340338713) q[2];
cx q[0],q[2];
ry(1.3244159908755675) q[0];
ry(-2.9765258251988684) q[2];
cx q[0],q[2];
ry(-2.3168158926378077) q[0];
ry(-2.3231232931429338) q[3];
cx q[0],q[3];
ry(-0.17514160493454795) q[0];
ry(-2.548324325110641) q[3];
cx q[0],q[3];
ry(-2.0613698369399724) q[1];
ry(0.596792064464478) q[2];
cx q[1],q[2];
ry(-1.5916377375135673) q[1];
ry(-2.3927724274124733) q[2];
cx q[1],q[2];
ry(2.347837663840487) q[1];
ry(0.9851470108094622) q[3];
cx q[1],q[3];
ry(1.2189557877583805) q[1];
ry(-2.0508621757269765) q[3];
cx q[1],q[3];
ry(2.2941050827168845) q[2];
ry(2.482217849454645) q[3];
cx q[2],q[3];
ry(-0.7750910535825968) q[2];
ry(-2.5179922062557205) q[3];
cx q[2],q[3];
ry(2.876896484577645) q[0];
ry(-1.1389994704060706) q[1];
ry(2.6746133902367117) q[2];
ry(-0.05991693520904175) q[3];