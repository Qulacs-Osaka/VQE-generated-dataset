OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.3819508114781525) q[0];
ry(1.802185060487476) q[1];
cx q[0],q[1];
ry(-1.351926867716163) q[0];
ry(1.6981690717477314) q[1];
cx q[0],q[1];
ry(2.4255476466523582) q[1];
ry(-0.6994006046432498) q[2];
cx q[1],q[2];
ry(-0.5605470659247481) q[1];
ry(-0.7194560384054443) q[2];
cx q[1],q[2];
ry(-0.41938746117443326) q[2];
ry(-2.8764664995015696) q[3];
cx q[2],q[3];
ry(-2.705427028043428) q[2];
ry(0.946873066456057) q[3];
cx q[2],q[3];
ry(-2.09342824617912) q[0];
ry(0.3939892142574218) q[1];
cx q[0],q[1];
ry(1.8937660609557674) q[0];
ry(1.918082693975162) q[1];
cx q[0],q[1];
ry(2.808046821937282) q[1];
ry(-0.8966273814288174) q[2];
cx q[1],q[2];
ry(-2.6948604033031143) q[1];
ry(-1.05614093818241) q[2];
cx q[1],q[2];
ry(-0.43919949623237375) q[2];
ry(-2.9674220051710902) q[3];
cx q[2],q[3];
ry(-0.25415233584431896) q[2];
ry(-2.7948964381780845) q[3];
cx q[2],q[3];
ry(-2.4367673462450066) q[0];
ry(-2.7015298689288385) q[1];
cx q[0],q[1];
ry(2.0039466298810624) q[0];
ry(0.16393769572448136) q[1];
cx q[0],q[1];
ry(2.547639680524463) q[1];
ry(-1.7030604652554666) q[2];
cx q[1],q[2];
ry(-3.132762205284104) q[1];
ry(-0.06607903085556632) q[2];
cx q[1],q[2];
ry(-2.8749400065633877) q[2];
ry(0.6774404384191913) q[3];
cx q[2],q[3];
ry(-0.005004715961067906) q[2];
ry(-2.644322475714522) q[3];
cx q[2],q[3];
ry(-3.0777962773869807) q[0];
ry(2.618525106951353) q[1];
cx q[0],q[1];
ry(2.8044680470905443) q[0];
ry(-0.18281864573117837) q[1];
cx q[0],q[1];
ry(-0.6890052093664915) q[1];
ry(1.0345036306174884) q[2];
cx q[1],q[2];
ry(-0.8015794965017458) q[1];
ry(-0.4729987215933251) q[2];
cx q[1],q[2];
ry(-0.2160412530772527) q[2];
ry(-1.02313038227253) q[3];
cx q[2],q[3];
ry(0.986261320965629) q[2];
ry(-0.6469852160873124) q[3];
cx q[2],q[3];
ry(-0.14775485533200267) q[0];
ry(3.113969999731966) q[1];
cx q[0],q[1];
ry(-0.40228784961811304) q[0];
ry(1.190618780250022) q[1];
cx q[0],q[1];
ry(1.5021701161445395) q[1];
ry(-1.948465874001946) q[2];
cx q[1],q[2];
ry(-1.9217727605208097) q[1];
ry(-1.7819977989656406) q[2];
cx q[1],q[2];
ry(-2.8945851351648715) q[2];
ry(1.708682099433224) q[3];
cx q[2],q[3];
ry(1.258086539079252) q[2];
ry(-1.8054464587702475) q[3];
cx q[2],q[3];
ry(-2.5850428233887057) q[0];
ry(1.4433461358890984) q[1];
cx q[0],q[1];
ry(-2.3838418335497003) q[0];
ry(0.9928253997863012) q[1];
cx q[0],q[1];
ry(1.384402700364359) q[1];
ry(-0.5145968424544569) q[2];
cx q[1],q[2];
ry(0.13145663581188494) q[1];
ry(0.8151774431216235) q[2];
cx q[1],q[2];
ry(0.582565868375859) q[2];
ry(-0.28760875892449433) q[3];
cx q[2],q[3];
ry(-0.3308787730571728) q[2];
ry(1.3810627265007531) q[3];
cx q[2],q[3];
ry(-0.3920528018648688) q[0];
ry(-0.41454820747716337) q[1];
cx q[0],q[1];
ry(2.1271047802135916) q[0];
ry(-2.7817580469542955) q[1];
cx q[0],q[1];
ry(-1.5538428338585915) q[1];
ry(1.6357653138735573) q[2];
cx q[1],q[2];
ry(2.337821709279333) q[1];
ry(2.528120170880655) q[2];
cx q[1],q[2];
ry(-2.2938460226550803) q[2];
ry(3.013445987815312) q[3];
cx q[2],q[3];
ry(-2.074256309648515) q[2];
ry(0.7535366553977338) q[3];
cx q[2],q[3];
ry(1.7000603461501025) q[0];
ry(-1.7002039629818606) q[1];
cx q[0],q[1];
ry(2.122785621941988) q[0];
ry(1.2576753124273399) q[1];
cx q[0],q[1];
ry(-1.4371164335685322) q[1];
ry(0.3089067593588881) q[2];
cx q[1],q[2];
ry(0.937000399660893) q[1];
ry(3.0744565833346598) q[2];
cx q[1],q[2];
ry(-0.8979291980191586) q[2];
ry(2.647439716015332) q[3];
cx q[2],q[3];
ry(-1.156504409738222) q[2];
ry(2.705018503816276) q[3];
cx q[2],q[3];
ry(0.08318471828472163) q[0];
ry(3.0132538955469204) q[1];
cx q[0],q[1];
ry(-3.0777181282050714) q[0];
ry(-2.4138133541379987) q[1];
cx q[0],q[1];
ry(-3.001408503171912) q[1];
ry(-2.0180815099522924) q[2];
cx q[1],q[2];
ry(2.3777617695418414) q[1];
ry(1.1576196556023675) q[2];
cx q[1],q[2];
ry(3.0116616498689317) q[2];
ry(2.679066517184531) q[3];
cx q[2],q[3];
ry(1.4714899652308777) q[2];
ry(-1.0397504291655721) q[3];
cx q[2],q[3];
ry(-2.772010359029636) q[0];
ry(-0.4027014883663729) q[1];
cx q[0],q[1];
ry(-2.3854296144819673) q[0];
ry(-0.08054615564560397) q[1];
cx q[0],q[1];
ry(0.6427104009416711) q[1];
ry(-2.466706339635444) q[2];
cx q[1],q[2];
ry(2.5830486574939466) q[1];
ry(1.0543905362471324) q[2];
cx q[1],q[2];
ry(2.1275192243631587) q[2];
ry(2.6024633907715433) q[3];
cx q[2],q[3];
ry(0.8218307211292062) q[2];
ry(-1.2120670020820592) q[3];
cx q[2],q[3];
ry(1.4160969599734212) q[0];
ry(-1.347775428951962) q[1];
cx q[0],q[1];
ry(-1.5472656071356565) q[0];
ry(2.7280360016977903) q[1];
cx q[0],q[1];
ry(-2.090812362702917) q[1];
ry(-1.8927078589519313) q[2];
cx q[1],q[2];
ry(2.9978074368058016) q[1];
ry(-2.7609844713843468) q[2];
cx q[1],q[2];
ry(0.4969139535326663) q[2];
ry(-2.6260406234533593) q[3];
cx q[2],q[3];
ry(-1.8045246807957689) q[2];
ry(-2.2717426474627747) q[3];
cx q[2],q[3];
ry(1.7760519863408843) q[0];
ry(-0.18758596687177853) q[1];
cx q[0],q[1];
ry(0.7727603220576974) q[0];
ry(-1.5259664551415995) q[1];
cx q[0],q[1];
ry(-0.7853277581859395) q[1];
ry(0.21203042336477207) q[2];
cx q[1],q[2];
ry(-1.9028613458186907) q[1];
ry(-0.8137128959829557) q[2];
cx q[1],q[2];
ry(1.5832524708217648) q[2];
ry(-0.29573045961533007) q[3];
cx q[2],q[3];
ry(2.8160697972530344) q[2];
ry(2.9790680687299274) q[3];
cx q[2],q[3];
ry(1.5340795988329052) q[0];
ry(1.0077941918841828) q[1];
cx q[0],q[1];
ry(-3.095317383673801) q[0];
ry(-1.4196144830108652) q[1];
cx q[0],q[1];
ry(0.8928430967516637) q[1];
ry(-2.250932358552584) q[2];
cx q[1],q[2];
ry(-0.2784535684377083) q[1];
ry(0.384855530670249) q[2];
cx q[1],q[2];
ry(-2.4713914837791777) q[2];
ry(-1.2121481767796878) q[3];
cx q[2],q[3];
ry(2.992106613229154) q[2];
ry(-1.9717969131097215) q[3];
cx q[2],q[3];
ry(0.9981056317720762) q[0];
ry(-0.4158759811955975) q[1];
cx q[0],q[1];
ry(-0.31019852202200693) q[0];
ry(2.368870864061487) q[1];
cx q[0],q[1];
ry(-2.5786461792350406) q[1];
ry(0.6892473252610866) q[2];
cx q[1],q[2];
ry(1.6369897362517052) q[1];
ry(0.49784146263382917) q[2];
cx q[1],q[2];
ry(-0.3112509351532514) q[2];
ry(-1.134285474496652) q[3];
cx q[2],q[3];
ry(2.4848394337251962) q[2];
ry(-1.5723546083508495) q[3];
cx q[2],q[3];
ry(-1.1627860081956356) q[0];
ry(1.209673222754147) q[1];
cx q[0],q[1];
ry(-2.999966102795912) q[0];
ry(-1.8475152174903506) q[1];
cx q[0],q[1];
ry(-0.2941346185247105) q[1];
ry(-1.6971267072472118) q[2];
cx q[1],q[2];
ry(2.642572199663252) q[1];
ry(-0.5164370354080241) q[2];
cx q[1],q[2];
ry(-1.8001987046574623) q[2];
ry(-1.42697159482014) q[3];
cx q[2],q[3];
ry(2.255059584121606) q[2];
ry(2.9776928792520243) q[3];
cx q[2],q[3];
ry(-0.5884295935816849) q[0];
ry(-1.6923940435932332) q[1];
cx q[0],q[1];
ry(1.7208314490861045) q[0];
ry(-0.8055778924022361) q[1];
cx q[0],q[1];
ry(2.6654771761721876) q[1];
ry(-1.7565007501071914) q[2];
cx q[1],q[2];
ry(0.13376653985621179) q[1];
ry(3.1204857807540534) q[2];
cx q[1],q[2];
ry(1.5055627248173176) q[2];
ry(-2.98054487700515) q[3];
cx q[2],q[3];
ry(-1.38584768697537) q[2];
ry(1.040099949632535) q[3];
cx q[2],q[3];
ry(-0.9950250767084334) q[0];
ry(2.7175842590418062) q[1];
cx q[0],q[1];
ry(0.031922077439818786) q[0];
ry(-1.0681190688489979) q[1];
cx q[0],q[1];
ry(2.566383073848703) q[1];
ry(0.25435069080385553) q[2];
cx q[1],q[2];
ry(-0.679876219102728) q[1];
ry(0.07296819940590016) q[2];
cx q[1],q[2];
ry(0.09760551117759508) q[2];
ry(-1.23655931406938) q[3];
cx q[2],q[3];
ry(-1.2287779220389043) q[2];
ry(2.2402315516601927) q[3];
cx q[2],q[3];
ry(-2.258076541627841) q[0];
ry(-1.9209574944906327) q[1];
cx q[0],q[1];
ry(1.812423374342939) q[0];
ry(1.9157088756256844) q[1];
cx q[0],q[1];
ry(3.13390259864357) q[1];
ry(-1.4896572254613742) q[2];
cx q[1],q[2];
ry(-2.534655818192792) q[1];
ry(0.7446869868695982) q[2];
cx q[1],q[2];
ry(-2.555033628168628) q[2];
ry(-1.5444503769395244) q[3];
cx q[2],q[3];
ry(-1.0418892313817238) q[2];
ry(0.3838766967829912) q[3];
cx q[2],q[3];
ry(0.9143890726962933) q[0];
ry(0.13592678012029674) q[1];
cx q[0],q[1];
ry(-1.1773148592217586) q[0];
ry(-2.123903995692902) q[1];
cx q[0],q[1];
ry(-1.4262465553358297) q[1];
ry(-2.9519758668849905) q[2];
cx q[1],q[2];
ry(-1.03568760744315) q[1];
ry(-3.104293989806419) q[2];
cx q[1],q[2];
ry(2.871914588132665) q[2];
ry(-2.6314839780108543) q[3];
cx q[2],q[3];
ry(1.2664496586238805) q[2];
ry(-1.3243896549061898) q[3];
cx q[2],q[3];
ry(2.421341269997956) q[0];
ry(2.1121474834272074) q[1];
cx q[0],q[1];
ry(2.621969159925105) q[0];
ry(-0.7302393221557395) q[1];
cx q[0],q[1];
ry(1.9589088980392955) q[1];
ry(2.783428460486916) q[2];
cx q[1],q[2];
ry(-1.340466367219275) q[1];
ry(-0.6267700639141864) q[2];
cx q[1],q[2];
ry(-0.9988301943547571) q[2];
ry(-0.41072918487500987) q[3];
cx q[2],q[3];
ry(0.8828462447748161) q[2];
ry(-1.2458775254246595) q[3];
cx q[2],q[3];
ry(-0.45401359212602) q[0];
ry(-0.88641784735637) q[1];
cx q[0],q[1];
ry(-1.9116082720678884) q[0];
ry(-2.770759637283498) q[1];
cx q[0],q[1];
ry(-1.7283567113317102) q[1];
ry(-1.1500589643740835) q[2];
cx q[1],q[2];
ry(0.9204983618396317) q[1];
ry(-0.29556657210857157) q[2];
cx q[1],q[2];
ry(1.7455771138022569) q[2];
ry(0.9847640251232174) q[3];
cx q[2],q[3];
ry(-0.5865794399289062) q[2];
ry(-2.5446852045349413) q[3];
cx q[2],q[3];
ry(0.9363786869019645) q[0];
ry(-1.6196953261122404) q[1];
cx q[0],q[1];
ry(2.447834214230313) q[0];
ry(-2.4054848715462795) q[1];
cx q[0],q[1];
ry(-1.629368497573825) q[1];
ry(-0.1700734794930404) q[2];
cx q[1],q[2];
ry(2.1482160210248784) q[1];
ry(2.4433737606633277) q[2];
cx q[1],q[2];
ry(-1.7171603407384994) q[2];
ry(-3.093670001163954) q[3];
cx q[2],q[3];
ry(1.0939721633212816) q[2];
ry(2.480796553340803) q[3];
cx q[2],q[3];
ry(1.3120895236242403) q[0];
ry(-0.3984918276063415) q[1];
cx q[0],q[1];
ry(-3.075621899265179) q[0];
ry(1.9201983798645141) q[1];
cx q[0],q[1];
ry(0.5967698629660881) q[1];
ry(0.047528747251949) q[2];
cx q[1],q[2];
ry(0.0954356369302225) q[1];
ry(2.401321555472905) q[2];
cx q[1],q[2];
ry(-2.494080883701909) q[2];
ry(-2.305845895453287) q[3];
cx q[2],q[3];
ry(-2.642345845795824) q[2];
ry(0.2613558475077431) q[3];
cx q[2],q[3];
ry(-0.9232014366393759) q[0];
ry(2.933890542328395) q[1];
cx q[0],q[1];
ry(1.295505291622553) q[0];
ry(0.5120442958609441) q[1];
cx q[0],q[1];
ry(-2.952251663711343) q[1];
ry(-3.1343703597392314) q[2];
cx q[1],q[2];
ry(1.6631887633170992) q[1];
ry(2.985401777302483) q[2];
cx q[1],q[2];
ry(0.7336781027508268) q[2];
ry(0.9459461662226069) q[3];
cx q[2],q[3];
ry(2.2548786670671874) q[2];
ry(2.285649986095024) q[3];
cx q[2],q[3];
ry(2.276093820913583) q[0];
ry(0.3653084207399475) q[1];
cx q[0],q[1];
ry(1.3848569405520141) q[0];
ry(2.3943964010646184) q[1];
cx q[0],q[1];
ry(-1.271767999331538) q[1];
ry(-2.8223154992578094) q[2];
cx q[1],q[2];
ry(-1.6970231809275171) q[1];
ry(-2.511807135784929) q[2];
cx q[1],q[2];
ry(-3.1203600889081917) q[2];
ry(-1.5436930443746049) q[3];
cx q[2],q[3];
ry(-2.6473326632859626) q[2];
ry(0.428356105033763) q[3];
cx q[2],q[3];
ry(2.6224647432979817) q[0];
ry(0.4297711234972279) q[1];
ry(1.6086790032157399) q[2];
ry(0.5041276295310685) q[3];