OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.9610152197425137) q[0];
ry(-2.882663749787805) q[1];
cx q[0],q[1];
ry(1.2685773767846853) q[0];
ry(-3.036436401658434) q[1];
cx q[0],q[1];
ry(-2.744677205533394) q[1];
ry(-1.6260869782595455) q[2];
cx q[1],q[2];
ry(2.7914441788300945) q[1];
ry(-2.934616119271526) q[2];
cx q[1],q[2];
ry(-0.9365500997721625) q[2];
ry(2.6692990548125013) q[3];
cx q[2],q[3];
ry(-1.8064259026445326) q[2];
ry(-0.8989869345301598) q[3];
cx q[2],q[3];
ry(-2.5539269201550066) q[3];
ry(2.9164570074668426) q[4];
cx q[3],q[4];
ry(2.752962956006401) q[3];
ry(-0.6310040063316267) q[4];
cx q[3],q[4];
ry(-3.025043086846816) q[4];
ry(3.0735775959816993) q[5];
cx q[4],q[5];
ry(0.031956857068323075) q[4];
ry(3.139820163624736) q[5];
cx q[4],q[5];
ry(2.0473144971927124) q[5];
ry(-2.7791649373069536) q[6];
cx q[5],q[6];
ry(0.8560627290433391) q[5];
ry(-2.3211253327230392) q[6];
cx q[5],q[6];
ry(-1.6748867581178954) q[6];
ry(-0.7511491782728151) q[7];
cx q[6],q[7];
ry(-2.2862809806878124) q[6];
ry(1.8058578934223861) q[7];
cx q[6],q[7];
ry(-1.8245504319686914) q[7];
ry(2.246210348878029) q[8];
cx q[7],q[8];
ry(2.440882325257882) q[7];
ry(3.13784404825215) q[8];
cx q[7],q[8];
ry(2.628406082576188) q[8];
ry(-2.5525706974015705) q[9];
cx q[8],q[9];
ry(3.0315705148204746) q[8];
ry(1.8458589412900053) q[9];
cx q[8],q[9];
ry(2.52898472097044) q[9];
ry(-2.8050788194882568) q[10];
cx q[9],q[10];
ry(1.5792688843104683) q[9];
ry(-0.4166902450142542) q[10];
cx q[9],q[10];
ry(1.438449923669002) q[10];
ry(-3.127140984391623) q[11];
cx q[10],q[11];
ry(1.5731496299036714) q[10];
ry(1.4594271998993236) q[11];
cx q[10],q[11];
ry(1.0069547160214039) q[11];
ry(2.6689653074487962) q[12];
cx q[11],q[12];
ry(-0.9877965404715897) q[11];
ry(1.3462150154966332) q[12];
cx q[11],q[12];
ry(-0.995558217943838) q[12];
ry(-1.5019642086227882) q[13];
cx q[12],q[13];
ry(2.863336999160948) q[12];
ry(-0.5201642091963906) q[13];
cx q[12],q[13];
ry(-0.8272149667009563) q[13];
ry(-1.2130759583601685) q[14];
cx q[13],q[14];
ry(1.4882758891995032) q[13];
ry(0.4037968188886681) q[14];
cx q[13],q[14];
ry(2.4178880793523865) q[14];
ry(0.9227763235462527) q[15];
cx q[14],q[15];
ry(-0.6780227788203592) q[14];
ry(-2.4677588057947224) q[15];
cx q[14],q[15];
ry(2.943558110963525) q[15];
ry(-2.1214350602144636) q[16];
cx q[15],q[16];
ry(-2.4304244104819697) q[15];
ry(-1.3151901705667404) q[16];
cx q[15],q[16];
ry(1.7079151328312516) q[16];
ry(2.2160717902548877) q[17];
cx q[16],q[17];
ry(3.1333075804489847) q[16];
ry(-0.0010759029958107291) q[17];
cx q[16],q[17];
ry(-0.21331041100714979) q[17];
ry(-2.077899489893972) q[18];
cx q[17],q[18];
ry(-1.9601983193977057) q[17];
ry(0.33119953520684187) q[18];
cx q[17],q[18];
ry(-1.8642227549571233) q[18];
ry(0.4743519974389132) q[19];
cx q[18],q[19];
ry(2.2717562003265623) q[18];
ry(0.09962479878794019) q[19];
cx q[18],q[19];
ry(-2.265175104050071) q[0];
ry(2.039114058949292) q[1];
cx q[0],q[1];
ry(1.6651991553950325) q[0];
ry(-3.0562368827752633) q[1];
cx q[0],q[1];
ry(2.2052356778336533) q[1];
ry(-0.5601700663129092) q[2];
cx q[1],q[2];
ry(-2.119646173839524) q[1];
ry(0.11379483556441593) q[2];
cx q[1],q[2];
ry(1.2761615701768572) q[2];
ry(-2.7004203923922803) q[3];
cx q[2],q[3];
ry(-1.7372203656185192) q[2];
ry(0.6228446989738634) q[3];
cx q[2],q[3];
ry(3.0635013730561984) q[3];
ry(-0.4810550702015557) q[4];
cx q[3],q[4];
ry(3.1152988037534244) q[3];
ry(0.5440753872500927) q[4];
cx q[3],q[4];
ry(0.9820632270456207) q[4];
ry(1.2163239859643014) q[5];
cx q[4],q[5];
ry(1.5455516087162107) q[4];
ry(3.1410318171319442) q[5];
cx q[4],q[5];
ry(0.0902911582781174) q[5];
ry(2.5571313689948405) q[6];
cx q[5],q[6];
ry(-3.115039550286832) q[5];
ry(-0.06982387127896494) q[6];
cx q[5],q[6];
ry(-2.3690626872810765) q[6];
ry(0.8824947623721554) q[7];
cx q[6],q[7];
ry(-1.9654455814004486) q[6];
ry(1.750821956329185) q[7];
cx q[6],q[7];
ry(0.5192784782999391) q[7];
ry(-2.7804297482246243) q[8];
cx q[7],q[8];
ry(-1.7144989586055566) q[7];
ry(-3.1117484241591273) q[8];
cx q[7],q[8];
ry(1.3341712225081421) q[8];
ry(0.23191872935082863) q[9];
cx q[8],q[9];
ry(1.3362321567366378) q[8];
ry(-0.7130659745521131) q[9];
cx q[8],q[9];
ry(-2.9965703066050327) q[9];
ry(-3.025566237703049) q[10];
cx q[9],q[10];
ry(-3.0891776365216885) q[9];
ry(0.2299625261692686) q[10];
cx q[9],q[10];
ry(2.437686062879546) q[10];
ry(1.192204644104077) q[11];
cx q[10],q[11];
ry(0.01519379062774373) q[10];
ry(-3.1410474591053115) q[11];
cx q[10],q[11];
ry(2.3463933520883056) q[11];
ry(1.4373775957896993) q[12];
cx q[11],q[12];
ry(0.7278562033590239) q[11];
ry(2.18533336841311) q[12];
cx q[11],q[12];
ry(1.4155685737158972) q[12];
ry(0.8736385458453572) q[13];
cx q[12],q[13];
ry(1.9463897493019378) q[12];
ry(-1.2776538400086288) q[13];
cx q[12],q[13];
ry(-2.5645407966661184) q[13];
ry(-1.4404029087369385) q[14];
cx q[13],q[14];
ry(-2.8603606064677267) q[13];
ry(-3.03443678609318) q[14];
cx q[13],q[14];
ry(1.5631241861981024) q[14];
ry(-2.2983635954689277) q[15];
cx q[14],q[15];
ry(1.724997645662472) q[14];
ry(-2.232805877079795) q[15];
cx q[14],q[15];
ry(2.8086830551178394) q[15];
ry(2.575299959558149) q[16];
cx q[15],q[16];
ry(-2.147064713254532) q[15];
ry(-2.2680036529955476) q[16];
cx q[15],q[16];
ry(-2.0754210800318256) q[16];
ry(-2.4317640616316796) q[17];
cx q[16],q[17];
ry(1.8248973545574798) q[16];
ry(0.021270715012147612) q[17];
cx q[16],q[17];
ry(-2.5167832250648687) q[17];
ry(2.3716083055702355) q[18];
cx q[17],q[18];
ry(0.1748594788550544) q[17];
ry(-2.1196425859446566) q[18];
cx q[17],q[18];
ry(0.28069885637463765) q[18];
ry(0.10578186533612667) q[19];
cx q[18],q[19];
ry(-0.41455107739084884) q[18];
ry(-2.652016355497027) q[19];
cx q[18],q[19];
ry(-0.2787995016817191) q[0];
ry(-0.23372647227461663) q[1];
cx q[0],q[1];
ry(-0.9549119268984724) q[0];
ry(-1.908170790961256) q[1];
cx q[0],q[1];
ry(-2.593353350466302) q[1];
ry(-0.0029267885767367834) q[2];
cx q[1],q[2];
ry(2.711527318747502) q[1];
ry(-3.014046080269361) q[2];
cx q[1],q[2];
ry(-2.6907328842073786) q[2];
ry(-1.9484501748549894) q[3];
cx q[2],q[3];
ry(1.4669666035631463) q[2];
ry(-2.4453937194623236) q[3];
cx q[2],q[3];
ry(2.120608459123351) q[3];
ry(-1.2315763441322753) q[4];
cx q[3],q[4];
ry(3.0640450098087135) q[3];
ry(2.7111799854497334) q[4];
cx q[3],q[4];
ry(1.54823218742911) q[4];
ry(-2.953280850714659) q[5];
cx q[4],q[5];
ry(-0.002487200307458082) q[4];
ry(-0.00680603608234609) q[5];
cx q[4],q[5];
ry(-1.1819377062661376) q[5];
ry(-0.23454725127353163) q[6];
cx q[5],q[6];
ry(0.817753461854994) q[5];
ry(-2.8966049632403967) q[6];
cx q[5],q[6];
ry(1.540133522465771) q[6];
ry(0.5697397525356083) q[7];
cx q[6],q[7];
ry(1.551171693138448) q[6];
ry(0.17173580988549095) q[7];
cx q[6],q[7];
ry(-1.5774085832164984) q[7];
ry(-1.6217376459542843) q[8];
cx q[7],q[8];
ry(2.604926266681762) q[7];
ry(-1.5807284176741652) q[8];
cx q[7],q[8];
ry(-1.5484821337857024) q[8];
ry(-1.6555678031572114) q[9];
cx q[8],q[9];
ry(-1.1175918168674421) q[8];
ry(0.26446766347436856) q[9];
cx q[8],q[9];
ry(-0.23115572499156922) q[9];
ry(-2.0431688608897822) q[10];
cx q[9],q[10];
ry(-1.2434842181847647) q[9];
ry(0.9595603093623746) q[10];
cx q[9],q[10];
ry(-3.030136293060546) q[10];
ry(-2.285688065651625) q[11];
cx q[10],q[11];
ry(3.128460692760161) q[10];
ry(-3.131100698964837) q[11];
cx q[10],q[11];
ry(0.2501698019071865) q[11];
ry(-3.1079489329948773) q[12];
cx q[11],q[12];
ry(-3.11051428833455) q[11];
ry(-1.9569985834640313) q[12];
cx q[11],q[12];
ry(-0.003388569335870805) q[12];
ry(-2.9252096176274756) q[13];
cx q[12],q[13];
ry(0.4316348941390743) q[12];
ry(-2.150873429915735) q[13];
cx q[12],q[13];
ry(-1.5578189554559876) q[13];
ry(-2.917829163187084) q[14];
cx q[13],q[14];
ry(-2.7075740492045406) q[13];
ry(-1.5652717767148259) q[14];
cx q[13],q[14];
ry(1.966311663940564) q[14];
ry(0.11494488205591137) q[15];
cx q[14],q[15];
ry(1.1639308337883199) q[14];
ry(0.24100050220574065) q[15];
cx q[14],q[15];
ry(-1.1451367531812195) q[15];
ry(0.14032160241479374) q[16];
cx q[15],q[16];
ry(3.0788458330151154) q[15];
ry(2.725990663653407) q[16];
cx q[15],q[16];
ry(1.8438348708306336) q[16];
ry(-1.846551462997363) q[17];
cx q[16],q[17];
ry(-1.7320758687557873) q[16];
ry(-0.025654213657025693) q[17];
cx q[16],q[17];
ry(2.2173936901942337) q[17];
ry(1.5046474862292534) q[18];
cx q[17],q[18];
ry(-1.5199145350481968) q[17];
ry(2.8921898279895966) q[18];
cx q[17],q[18];
ry(-0.038254409732809816) q[18];
ry(-1.7570361696071914) q[19];
cx q[18],q[19];
ry(-1.81521679347503) q[18];
ry(1.3530201662913952) q[19];
cx q[18],q[19];
ry(-1.617576235521014) q[0];
ry(1.415434189849507) q[1];
cx q[0],q[1];
ry(-0.4361500866449988) q[0];
ry(3.11667826164718) q[1];
cx q[0],q[1];
ry(0.2162960542192933) q[1];
ry(2.64957147450643) q[2];
cx q[1],q[2];
ry(3.140610867551575) q[1];
ry(-1.019185465805962) q[2];
cx q[1],q[2];
ry(-2.4260772832492417) q[2];
ry(-2.6215364706262467) q[3];
cx q[2],q[3];
ry(2.4980494807114746) q[2];
ry(-2.02118002858915) q[3];
cx q[2],q[3];
ry(-0.017282336998426295) q[3];
ry(-1.5968489878629333) q[4];
cx q[3],q[4];
ry(-1.7701859476841317) q[3];
ry(-2.681932215563678) q[4];
cx q[3],q[4];
ry(-0.5724116542833435) q[4];
ry(0.9562108339433841) q[5];
cx q[4],q[5];
ry(3.1406633613844774) q[4];
ry(-3.1406783452164984) q[5];
cx q[4],q[5];
ry(1.9117708893156409) q[5];
ry(-1.6027961172051735) q[6];
cx q[5],q[6];
ry(-1.5457788428667314) q[5];
ry(1.4541677161817192) q[6];
cx q[5],q[6];
ry(-1.5622036737138094) q[6];
ry(-1.593038546428997) q[7];
cx q[6],q[7];
ry(-2.320872781593702) q[6];
ry(-1.575509420267916) q[7];
cx q[6],q[7];
ry(-2.0113824578027035) q[7];
ry(1.566653281206874) q[8];
cx q[7],q[8];
ry(1.558755383621165) q[7];
ry(2.9158551425705794) q[8];
cx q[7],q[8];
ry(-1.3788197054677367) q[8];
ry(1.2096534241229282) q[9];
cx q[8],q[9];
ry(2.6720763933656997) q[8];
ry(3.048009582386354) q[9];
cx q[8],q[9];
ry(-0.6704490767601063) q[9];
ry(-2.4363697068346712) q[10];
cx q[9],q[10];
ry(1.8088431109648768) q[9];
ry(0.8799509975276327) q[10];
cx q[9],q[10];
ry(-0.3072968988116047) q[10];
ry(1.5314576094753418) q[11];
cx q[10],q[11];
ry(3.134007487932127) q[10];
ry(3.0191477440276673) q[11];
cx q[10],q[11];
ry(0.5622618518470522) q[11];
ry(1.988934789657565) q[12];
cx q[11],q[12];
ry(3.1178314041772572) q[11];
ry(-3.0854185336042153) q[12];
cx q[11],q[12];
ry(1.9915169386054234) q[12];
ry(-1.3425520883137432) q[13];
cx q[12],q[13];
ry(2.0200215697332036) q[12];
ry(-3.1373773573313932) q[13];
cx q[12],q[13];
ry(2.571118293175976) q[13];
ry(2.9826872162032103) q[14];
cx q[13],q[14];
ry(0.23011568117202333) q[13];
ry(0.2243161540557373) q[14];
cx q[13],q[14];
ry(0.9065170377839156) q[14];
ry(-1.4960112694531524) q[15];
cx q[14],q[15];
ry(1.0683278747023008) q[14];
ry(2.8488616571070575) q[15];
cx q[14],q[15];
ry(2.359678554493467) q[15];
ry(0.6727805054563973) q[16];
cx q[15],q[16];
ry(0.9928409253463861) q[15];
ry(0.22410718809043836) q[16];
cx q[15],q[16];
ry(-1.0655950974443444) q[16];
ry(-2.104703786502937) q[17];
cx q[16],q[17];
ry(-0.23597757630039665) q[16];
ry(3.1245938705844964) q[17];
cx q[16],q[17];
ry(0.03668334142562291) q[17];
ry(0.36606915334925016) q[18];
cx q[17],q[18];
ry(-1.5074680966981058) q[17];
ry(-0.48180706231033615) q[18];
cx q[17],q[18];
ry(0.2101297602607977) q[18];
ry(3.136722946919605) q[19];
cx q[18],q[19];
ry(-1.886419555750006) q[18];
ry(0.9182105231069003) q[19];
cx q[18],q[19];
ry(3.0296295958424326) q[0];
ry(-1.3381769039152172) q[1];
cx q[0],q[1];
ry(1.897305139118859) q[0];
ry(2.249442034429423) q[1];
cx q[0],q[1];
ry(-2.3072597189719475) q[1];
ry(0.07727884613028338) q[2];
cx q[1],q[2];
ry(0.8250899384606522) q[1];
ry(1.1778648119058726) q[2];
cx q[1],q[2];
ry(2.0356182346907454) q[2];
ry(-1.7946839853911014) q[3];
cx q[2],q[3];
ry(1.9613759995980466) q[2];
ry(2.183574629318688) q[3];
cx q[2],q[3];
ry(0.7333330109059557) q[3];
ry(-0.9117113918960946) q[4];
cx q[3],q[4];
ry(-0.31470507045471674) q[3];
ry(-0.38605942740909693) q[4];
cx q[3],q[4];
ry(2.174539265382398) q[4];
ry(0.5935371228046273) q[5];
cx q[4],q[5];
ry(-0.00017239217119342933) q[4];
ry(-3.1092081952062713) q[5];
cx q[4],q[5];
ry(0.4802388051914998) q[5];
ry(-0.3737958606307736) q[6];
cx q[5],q[6];
ry(1.5038070239071728) q[5];
ry(1.4612255999576034) q[6];
cx q[5],q[6];
ry(-0.9561020172721143) q[6];
ry(-2.382820581753439) q[7];
cx q[6],q[7];
ry(-3.1177167859561754) q[6];
ry(-3.1309129683717973) q[7];
cx q[6],q[7];
ry(0.9805822697850379) q[7];
ry(-1.4133088299687953) q[8];
cx q[7],q[8];
ry(-2.9032802015730277) q[7];
ry(0.06958310619870982) q[8];
cx q[7],q[8];
ry(-0.2202089467732469) q[8];
ry(0.7873969258671762) q[9];
cx q[8],q[9];
ry(0.5756918385473021) q[8];
ry(-0.6585544831770352) q[9];
cx q[8],q[9];
ry(3.0631436913367134) q[9];
ry(-1.6934592674063085) q[10];
cx q[9],q[10];
ry(0.5523776500810973) q[9];
ry(-0.5086562971689196) q[10];
cx q[9],q[10];
ry(-1.5637949047627595) q[10];
ry(0.21481832869683082) q[11];
cx q[10],q[11];
ry(3.1010867816435104) q[10];
ry(-2.7680434850694122) q[11];
cx q[10],q[11];
ry(1.4016843986261567) q[11];
ry(-3.0744139420063434) q[12];
cx q[11],q[12];
ry(-2.6769466462952782) q[11];
ry(-1.5093030867137744) q[12];
cx q[11],q[12];
ry(0.014489689001549207) q[12];
ry(1.9826893479053094) q[13];
cx q[12],q[13];
ry(1.5514783400195011) q[12];
ry(1.5625056274256668) q[13];
cx q[12],q[13];
ry(2.7673600209239435) q[13];
ry(2.001051154280372) q[14];
cx q[13],q[14];
ry(0.15065677392731036) q[13];
ry(-1.5531210019134427) q[14];
cx q[13],q[14];
ry(2.696949312138333) q[14];
ry(2.2818859464868524) q[15];
cx q[14],q[15];
ry(-2.143195646899687) q[14];
ry(-0.1637389453189817) q[15];
cx q[14],q[15];
ry(1.1557734177278016) q[15];
ry(-3.0181485367691985) q[16];
cx q[15],q[16];
ry(0.11875125495298633) q[15];
ry(-0.2572586461746071) q[16];
cx q[15],q[16];
ry(1.9442840749792403) q[16];
ry(2.439188104816415) q[17];
cx q[16],q[17];
ry(-0.525822778392317) q[16];
ry(-3.138800660954742) q[17];
cx q[16],q[17];
ry(-2.7206364845222826) q[17];
ry(1.3506394797218648) q[18];
cx q[17],q[18];
ry(0.9525522083146001) q[17];
ry(1.3177835702657508) q[18];
cx q[17],q[18];
ry(3.1347542794360486) q[18];
ry(-1.2200330167347753) q[19];
cx q[18],q[19];
ry(0.43216951717095337) q[18];
ry(0.43877691391024864) q[19];
cx q[18],q[19];
ry(2.5238646080518685) q[0];
ry(2.520412283031615) q[1];
cx q[0],q[1];
ry(-1.0868811389432707) q[0];
ry(-0.19863092066598753) q[1];
cx q[0],q[1];
ry(-2.2471754923473823) q[1];
ry(-0.003726125998510119) q[2];
cx q[1],q[2];
ry(0.3592826911683122) q[1];
ry(2.9688275567535216) q[2];
cx q[1],q[2];
ry(0.27197777738768014) q[2];
ry(2.339369666225351) q[3];
cx q[2],q[3];
ry(2.032850737594023) q[2];
ry(-1.3673033412066609) q[3];
cx q[2],q[3];
ry(-1.4116357272215627) q[3];
ry(-1.3535012100744126) q[4];
cx q[3],q[4];
ry(1.5924493280726955) q[3];
ry(0.15442610314979885) q[4];
cx q[3],q[4];
ry(1.4923578115559397) q[4];
ry(0.13729680761154234) q[5];
cx q[4],q[5];
ry(1.5870826528852149) q[4];
ry(1.5704785715830196) q[5];
cx q[4],q[5];
ry(-3.038136590176683) q[5];
ry(-0.9144533031541648) q[6];
cx q[5],q[6];
ry(3.0625807871214468) q[5];
ry(-3.1334251366882473) q[6];
cx q[5],q[6];
ry(-1.464182254014963) q[6];
ry(1.1176777817309471) q[7];
cx q[6],q[7];
ry(-0.015443155054450335) q[6];
ry(3.11297220711105) q[7];
cx q[6],q[7];
ry(2.704963916435727) q[7];
ry(0.46437173991535635) q[8];
cx q[7],q[8];
ry(-0.09846116425420302) q[7];
ry(-0.06029895656015505) q[8];
cx q[7],q[8];
ry(2.8014239860449006) q[8];
ry(-0.6112376626616535) q[9];
cx q[8],q[9];
ry(-0.022922582291379) q[8];
ry(-0.5092417279846058) q[9];
cx q[8],q[9];
ry(2.3164360330369926) q[9];
ry(1.4574436768124723) q[10];
cx q[9],q[10];
ry(2.9381769981683017) q[9];
ry(-2.2163139382342063) q[10];
cx q[9],q[10];
ry(-1.837494824902354) q[10];
ry(-3.10871717117815) q[11];
cx q[10],q[11];
ry(1.2997718650113594) q[10];
ry(-3.1410097145326503) q[11];
cx q[10],q[11];
ry(0.7769119940815772) q[11];
ry(-0.006136508777061067) q[12];
cx q[11],q[12];
ry(-0.370923745017107) q[11];
ry(3.1234734080945463) q[12];
cx q[11],q[12];
ry(1.5876862881969498) q[12];
ry(-3.1406428220180986) q[13];
cx q[12],q[13];
ry(2.892736664915347) q[12];
ry(1.5762173742339562) q[13];
cx q[12],q[13];
ry(-1.6439358460500728) q[13];
ry(-1.764275852573176) q[14];
cx q[13],q[14];
ry(0.03266680294978741) q[13];
ry(-2.26725003230392) q[14];
cx q[13],q[14];
ry(-0.4902287863858661) q[14];
ry(1.2239284682665925) q[15];
cx q[14],q[15];
ry(0.46335442305092656) q[14];
ry(1.6997214023752552) q[15];
cx q[14],q[15];
ry(-0.6429828919771701) q[15];
ry(0.6626881442259717) q[16];
cx q[15],q[16];
ry(0.15217909057074977) q[15];
ry(-2.7958653694079594) q[16];
cx q[15],q[16];
ry(-2.907386478924039) q[16];
ry(-1.715962142331258) q[17];
cx q[16],q[17];
ry(2.084336371496511) q[16];
ry(-0.11222925401543979) q[17];
cx q[16],q[17];
ry(2.8697376882245544) q[17];
ry(1.2012831648056483) q[18];
cx q[17],q[18];
ry(-2.592992645222744) q[17];
ry(0.5534505788088584) q[18];
cx q[17],q[18];
ry(-1.8878100783288203) q[18];
ry(-2.0677481615996127) q[19];
cx q[18],q[19];
ry(-1.820539560122847) q[18];
ry(-0.5700227492041616) q[19];
cx q[18],q[19];
ry(-0.8482815137023794) q[0];
ry(-3.079991993514108) q[1];
cx q[0],q[1];
ry(3.0928260555755065) q[0];
ry(-3.0934829041419296) q[1];
cx q[0],q[1];
ry(2.7059395482281285) q[1];
ry(-0.4089495454096404) q[2];
cx q[1],q[2];
ry(-2.9490013236192802) q[1];
ry(-1.7436338209287616) q[2];
cx q[1],q[2];
ry(0.26892109967283595) q[2];
ry(1.22647752187436) q[3];
cx q[2],q[3];
ry(0.21230022619506708) q[2];
ry(2.5605362312185043) q[3];
cx q[2],q[3];
ry(-0.26183278889511463) q[3];
ry(-3.131636601187913) q[4];
cx q[3],q[4];
ry(1.5636866051384795) q[3];
ry(3.1364882183426412) q[4];
cx q[3],q[4];
ry(1.564200636348673) q[4];
ry(1.3916200309443578) q[5];
cx q[4],q[5];
ry(-1.444543505832663) q[4];
ry(-2.039299058631056) q[5];
cx q[4],q[5];
ry(2.7824176539108887) q[5];
ry(0.9929542648909386) q[6];
cx q[5],q[6];
ry(-2.600420115641) q[5];
ry(-1.183160750892854) q[6];
cx q[5],q[6];
ry(-1.7607768396085532) q[6];
ry(-2.150802878516801) q[7];
cx q[6],q[7];
ry(-0.0008883275520155111) q[6];
ry(-0.007459379509939397) q[7];
cx q[6],q[7];
ry(1.2501615352012774) q[7];
ry(-3.107801940746179) q[8];
cx q[7],q[8];
ry(1.9394986372208558) q[7];
ry(0.6619302307429393) q[8];
cx q[7],q[8];
ry(-1.7423103889393405) q[8];
ry(0.63630456256409) q[9];
cx q[8],q[9];
ry(0.028118044163027333) q[8];
ry(3.0879092825839423) q[9];
cx q[8],q[9];
ry(2.2076695628749463) q[9];
ry(-1.3908702402085602) q[10];
cx q[9],q[10];
ry(-2.335608890899113) q[9];
ry(0.6549686723448218) q[10];
cx q[9],q[10];
ry(-2.1877823943467396) q[10];
ry(-0.2545722305031851) q[11];
cx q[10],q[11];
ry(-3.1307423517442827) q[10];
ry(0.0055951053438006515) q[11];
cx q[10],q[11];
ry(2.4273827182447567) q[11];
ry(1.6240400371164372) q[12];
cx q[11],q[12];
ry(-0.27429963314596595) q[11];
ry(-0.029861354244482795) q[12];
cx q[11],q[12];
ry(0.4138947149764729) q[12];
ry(1.5970414565488262) q[13];
cx q[12],q[13];
ry(-2.90641402272312) q[12];
ry(0.01934097311885541) q[13];
cx q[12],q[13];
ry(-1.4629247347507033) q[13];
ry(1.6595126094666894) q[14];
cx q[13],q[14];
ry(-1.5380871925687583) q[13];
ry(1.5657093973637881) q[14];
cx q[13],q[14];
ry(-1.4097039658919848) q[14];
ry(-0.08490245248792946) q[15];
cx q[14],q[15];
ry(-0.013904559011165505) q[14];
ry(2.38549318228907) q[15];
cx q[14],q[15];
ry(2.42415837111662) q[15];
ry(2.342631674161711) q[16];
cx q[15],q[16];
ry(2.7983216260026795) q[15];
ry(-0.16524187990314862) q[16];
cx q[15],q[16];
ry(2.5840701656498197) q[16];
ry(-0.5135819524855609) q[17];
cx q[16],q[17];
ry(0.42501948263009476) q[16];
ry(2.715297468332927) q[17];
cx q[16],q[17];
ry(2.6005980836115863) q[17];
ry(1.6392049074352875) q[18];
cx q[17],q[18];
ry(1.7104497090739856) q[17];
ry(-1.4425924316527166) q[18];
cx q[17],q[18];
ry(-0.28853555996623015) q[18];
ry(1.0113116213126876) q[19];
cx q[18],q[19];
ry(-0.7554903909088536) q[18];
ry(-1.548940807547009) q[19];
cx q[18],q[19];
ry(2.2042859432691446) q[0];
ry(-3.0692600542069024) q[1];
cx q[0],q[1];
ry(1.6488179873176787) q[0];
ry(1.2799416055021275) q[1];
cx q[0],q[1];
ry(2.9526236455154087) q[1];
ry(-2.2655504616339197) q[2];
cx q[1],q[2];
ry(-0.9815682313289902) q[1];
ry(-2.7196493049170734) q[2];
cx q[1],q[2];
ry(-1.7068797975546266) q[2];
ry(-2.971610136157948) q[3];
cx q[2],q[3];
ry(-0.06002941413364441) q[2];
ry(1.9314019094978248) q[3];
cx q[2],q[3];
ry(1.4277047433862498) q[3];
ry(-2.4790554945912593) q[4];
cx q[3],q[4];
ry(3.1069021132201584) q[3];
ry(-0.001531660562453574) q[4];
cx q[3],q[4];
ry(2.9659434285253967) q[4];
ry(1.553336485486539) q[5];
cx q[4],q[5];
ry(0.34988225345356616) q[4];
ry(-3.132659001906161) q[5];
cx q[4],q[5];
ry(-0.5228212219755645) q[5];
ry(1.8061851555417572) q[6];
cx q[5],q[6];
ry(2.6468922923278684) q[5];
ry(1.2583468634592272) q[6];
cx q[5],q[6];
ry(-1.0781494973268968) q[6];
ry(-0.7246478218356707) q[7];
cx q[6],q[7];
ry(-3.1288747388938267) q[6];
ry(-3.1340122328621542) q[7];
cx q[6],q[7];
ry(2.3756197191841766) q[7];
ry(-1.7429394772480746) q[8];
cx q[7],q[8];
ry(-0.34578341730390333) q[7];
ry(0.740230275807984) q[8];
cx q[7],q[8];
ry(-0.43422054578317154) q[8];
ry(-1.2789853151315536) q[9];
cx q[8],q[9];
ry(1.7117280440525446) q[8];
ry(0.007857051356666117) q[9];
cx q[8],q[9];
ry(1.5983332701672557) q[9];
ry(0.9714376650550273) q[10];
cx q[9],q[10];
ry(1.5919825546326862) q[9];
ry(-2.824932553981732) q[10];
cx q[9],q[10];
ry(-0.07882357660476202) q[10];
ry(1.3098770035366316) q[11];
cx q[10],q[11];
ry(-0.0010012210214291513) q[10];
ry(1.5791310198878836) q[11];
cx q[10],q[11];
ry(-1.5904674864265083) q[11];
ry(-1.9541278441433945) q[12];
cx q[11],q[12];
ry(1.5950435883996916) q[11];
ry(-1.5893810166060245) q[12];
cx q[11],q[12];
ry(1.6363244621829787) q[12];
ry(-1.539248000807453) q[13];
cx q[12],q[13];
ry(-1.569758940852327) q[12];
ry(-0.025183979167423765) q[13];
cx q[12],q[13];
ry(-1.6090715234286885) q[13];
ry(-2.4191802796699045) q[14];
cx q[13],q[14];
ry(-2.8926971510139423) q[13];
ry(-0.51385792041789) q[14];
cx q[13],q[14];
ry(1.0438056644439673) q[14];
ry(-0.5634370207788278) q[15];
cx q[14],q[15];
ry(3.073957064064599) q[14];
ry(-2.8602859212649068) q[15];
cx q[14],q[15];
ry(-0.43697935396902376) q[15];
ry(-0.017537300395395732) q[16];
cx q[15],q[16];
ry(-1.5683199203709606) q[15];
ry(-1.4665756117265802) q[16];
cx q[15],q[16];
ry(-2.67637302264719) q[16];
ry(1.579489040616112) q[17];
cx q[16],q[17];
ry(1.504598465064227) q[16];
ry(1.5798319014333428) q[17];
cx q[16],q[17];
ry(2.9469933715964363) q[17];
ry(-1.503420748679887) q[18];
cx q[17],q[18];
ry(1.4281268374547835) q[17];
ry(-2.8007287786651194) q[18];
cx q[17],q[18];
ry(-0.00790742783795129) q[18];
ry(1.7941178536727551) q[19];
cx q[18],q[19];
ry(1.5661474462156744) q[18];
ry(1.1537926846229514) q[19];
cx q[18],q[19];
ry(0.7807821342812179) q[0];
ry(-3.040135520273606) q[1];
cx q[0],q[1];
ry(-2.6619573350104417) q[0];
ry(-1.3591844057634759) q[1];
cx q[0],q[1];
ry(-1.6676191771638829) q[1];
ry(2.272443213353484) q[2];
cx q[1],q[2];
ry(0.6309656041886607) q[1];
ry(-2.0244486092404337) q[2];
cx q[1],q[2];
ry(1.1291267209900588) q[2];
ry(-1.8113468489285118) q[3];
cx q[2],q[3];
ry(2.7752279203122474) q[2];
ry(-0.9071223656185693) q[3];
cx q[2],q[3];
ry(2.5968703382067364) q[3];
ry(-2.390719389455477) q[4];
cx q[3],q[4];
ry(1.0433219106103506) q[3];
ry(3.1288513249511323) q[4];
cx q[3],q[4];
ry(-2.59777159681348) q[4];
ry(3.0854550996333594) q[5];
cx q[4],q[5];
ry(-0.08186926448936482) q[4];
ry(-3.129892664310199) q[5];
cx q[4],q[5];
ry(2.1830317174146456) q[5];
ry(0.5864223095369336) q[6];
cx q[5],q[6];
ry(-3.0903120620865296) q[5];
ry(-1.921668105627745) q[6];
cx q[5],q[6];
ry(-2.4810354941734127) q[6];
ry(-2.1502034855602874) q[7];
cx q[6],q[7];
ry(0.009712344557875062) q[6];
ry(-3.1358650214834216) q[7];
cx q[6],q[7];
ry(-1.0207864095174823) q[7];
ry(2.767312724347523) q[8];
cx q[7],q[8];
ry(3.1101295511952793) q[7];
ry(-1.3681473540395466) q[8];
cx q[7],q[8];
ry(-1.5589552570747989) q[8];
ry(-0.8068039531783954) q[9];
cx q[8],q[9];
ry(3.124644314695111) q[8];
ry(0.12378714314107953) q[9];
cx q[8],q[9];
ry(0.310733927244204) q[9];
ry(1.5808851978823497) q[10];
cx q[9],q[10];
ry(1.9167913411692243) q[9];
ry(3.134816816667438) q[10];
cx q[9],q[10];
ry(-1.5514412961317863) q[10];
ry(-3.1366631872035122) q[11];
cx q[10],q[11];
ry(0.38798945357008424) q[10];
ry(-1.5320552108874947) q[11];
cx q[10],q[11];
ry(3.1329999557565085) q[11];
ry(-1.7956245511065876) q[12];
cx q[11],q[12];
ry(-2.495622262144012) q[11];
ry(-1.5642032065566387) q[12];
cx q[11],q[12];
ry(-1.5810174701703004) q[12];
ry(2.5338578670723493) q[13];
cx q[12],q[13];
ry(0.037298424333078735) q[12];
ry(1.4258314793057645) q[13];
cx q[12],q[13];
ry(0.6246085182604784) q[13];
ry(0.2863020970172139) q[14];
cx q[13],q[14];
ry(0.06569600521934249) q[13];
ry(-2.2515640049982455) q[14];
cx q[13],q[14];
ry(-2.6952644054254384) q[14];
ry(0.2899965802312181) q[15];
cx q[14],q[15];
ry(0.04443002222792547) q[14];
ry(0.02065067956493305) q[15];
cx q[14],q[15];
ry(1.2793793082572185) q[15];
ry(-1.914128559939397) q[16];
cx q[15],q[16];
ry(3.059203677410704) q[15];
ry(1.5564944699735346) q[16];
cx q[15],q[16];
ry(1.2618146674786717) q[16];
ry(-1.5589093968798462) q[17];
cx q[16],q[17];
ry(1.5779100116127118) q[16];
ry(-1.9538151268631125) q[17];
cx q[16],q[17];
ry(-1.5302439549353393) q[17];
ry(-3.0273015202662763) q[18];
cx q[17],q[18];
ry(3.113400039041544) q[17];
ry(2.7982167365564137) q[18];
cx q[17],q[18];
ry(-2.97117724835691) q[18];
ry(-1.7075331433432452) q[19];
cx q[18],q[19];
ry(1.1595938856061112) q[18];
ry(1.7599760675058942) q[19];
cx q[18],q[19];
ry(1.9405795975771842) q[0];
ry(-1.6496321115087285) q[1];
cx q[0],q[1];
ry(1.8879018875090194) q[0];
ry(-0.4639778723798782) q[1];
cx q[0],q[1];
ry(-2.8089740470234728) q[1];
ry(3.0743644390218976) q[2];
cx q[1],q[2];
ry(-0.0008459598843906235) q[1];
ry(-1.6006508686131404) q[2];
cx q[1],q[2];
ry(3.1076923767841467) q[2];
ry(-1.442602659169558) q[3];
cx q[2],q[3];
ry(0.3846741582478238) q[2];
ry(-1.3069284975151532) q[3];
cx q[2],q[3];
ry(-0.6631672188616632) q[3];
ry(0.6390796848106762) q[4];
cx q[3],q[4];
ry(0.2799334283159416) q[3];
ry(-0.004944883724503521) q[4];
cx q[3],q[4];
ry(-1.652683889950623) q[4];
ry(0.8635440390051139) q[5];
cx q[4],q[5];
ry(1.8324052324033755) q[4];
ry(0.687240278162662) q[5];
cx q[4],q[5];
ry(-2.8879783417694984) q[5];
ry(0.2758424068732621) q[6];
cx q[5],q[6];
ry(-2.99681897973019) q[5];
ry(-0.06153109000491815) q[6];
cx q[5],q[6];
ry(-1.4746120519392198) q[6];
ry(1.1821034964677297) q[7];
cx q[6],q[7];
ry(-2.0506056669470807) q[6];
ry(-0.6365801140368559) q[7];
cx q[6],q[7];
ry(0.45358951591124175) q[7];
ry(0.848315768445147) q[8];
cx q[7],q[8];
ry(0.3506395967521947) q[7];
ry(3.141096479775449) q[8];
cx q[7],q[8];
ry(-1.5684769688431797) q[8];
ry(1.5038686119470084) q[9];
cx q[8],q[9];
ry(-3.1275205149943726) q[8];
ry(-0.14973601165531747) q[9];
cx q[8],q[9];
ry(-0.6842387423815603) q[9];
ry(1.4563200176915185) q[10];
cx q[9],q[10];
ry(3.093712846817766) q[9];
ry(0.006650584046171027) q[10];
cx q[9],q[10];
ry(-3.0957918953512213) q[10];
ry(-1.34315618941381) q[11];
cx q[10],q[11];
ry(3.1394838289400324) q[10];
ry(-1.5614425040744357) q[11];
cx q[10],q[11];
ry(0.3112467185936331) q[11];
ry(1.7470641509897202) q[12];
cx q[11],q[12];
ry(0.6573545022294631) q[11];
ry(-3.130799159540674) q[12];
cx q[11],q[12];
ry(2.982741754608872) q[12];
ry(0.2774826234770691) q[13];
cx q[12],q[13];
ry(-3.085896991304343) q[12];
ry(-1.5389939819357132) q[13];
cx q[12],q[13];
ry(0.2645822835457613) q[13];
ry(-0.30729838653652713) q[14];
cx q[13],q[14];
ry(-1.8671040556092198) q[13];
ry(-1.5326722754479247) q[14];
cx q[13],q[14];
ry(-0.08852736950326089) q[14];
ry(3.0608444077077577) q[15];
cx q[14],q[15];
ry(0.36997105289732435) q[14];
ry(-0.3680313098797825) q[15];
cx q[14],q[15];
ry(1.632851297155975) q[15];
ry(1.6036449454743185) q[16];
cx q[15],q[16];
ry(-1.5508190621998548) q[15];
ry(1.5589265357473945) q[16];
cx q[15],q[16];
ry(1.5655845508795068) q[16];
ry(-3.036788383995234) q[17];
cx q[16],q[17];
ry(-1.5227871641334803) q[16];
ry(1.505077846608712) q[17];
cx q[16],q[17];
ry(-1.7262258539272182) q[17];
ry(1.8507837789158585) q[18];
cx q[17],q[18];
ry(-2.5231582841752234) q[17];
ry(1.6343578933253982) q[18];
cx q[17],q[18];
ry(-1.340307186223151) q[18];
ry(0.1846242156877998) q[19];
cx q[18],q[19];
ry(2.346482733533746) q[18];
ry(-2.901829750971467) q[19];
cx q[18],q[19];
ry(0.2784052597006639) q[0];
ry(2.2674418862036028) q[1];
cx q[0],q[1];
ry(2.973818571600726) q[0];
ry(-1.6419643420133738) q[1];
cx q[0],q[1];
ry(1.0027004988851624) q[1];
ry(-0.0994929554865438) q[2];
cx q[1],q[2];
ry(3.000899536041177) q[1];
ry(-1.1875571734738115) q[2];
cx q[1],q[2];
ry(2.8052768870580223) q[2];
ry(-1.606823665701178) q[3];
cx q[2],q[3];
ry(-0.06149446836483176) q[2];
ry(1.6708967330727704) q[3];
cx q[2],q[3];
ry(-2.929511007901652) q[3];
ry(-1.5888111383946981) q[4];
cx q[3],q[4];
ry(2.3239892217198115) q[3];
ry(3.1391557365028975) q[4];
cx q[3],q[4];
ry(-1.5280870332008227) q[4];
ry(-1.8690019140257224) q[5];
cx q[4],q[5];
ry(-1.5533820509450913) q[4];
ry(0.4462364493959905) q[5];
cx q[4],q[5];
ry(-2.4368522685483) q[5];
ry(-0.6528476584617504) q[6];
cx q[5],q[6];
ry(-0.005946875631362758) q[5];
ry(-0.008471586421238795) q[6];
cx q[5],q[6];
ry(-0.4321834482765084) q[6];
ry(0.4764395737523031) q[7];
cx q[6],q[7];
ry(2.628932794865032) q[6];
ry(1.2349318293211677) q[7];
cx q[6],q[7];
ry(-1.5527254207996375) q[7];
ry(1.785554674698714) q[8];
cx q[7],q[8];
ry(-0.32431300024923004) q[7];
ry(1.5683986007132322) q[8];
cx q[7],q[8];
ry(-1.2055217378681402) q[8];
ry(2.5798378948051788) q[9];
cx q[8],q[9];
ry(-2.3745087141916628) q[8];
ry(-0.008767991766547745) q[9];
cx q[8],q[9];
ry(-1.815269569812152) q[9];
ry(-3.123834393093487) q[10];
cx q[9],q[10];
ry(1.7659186250734358) q[9];
ry(3.1199282875020424) q[10];
cx q[9],q[10];
ry(2.4603721312976012) q[10];
ry(1.6807277147520034) q[11];
cx q[10],q[11];
ry(-1.8510242398927588) q[10];
ry(3.083170344590507) q[11];
cx q[10],q[11];
ry(1.5842826776020305) q[11];
ry(2.7641055532126715) q[12];
cx q[11],q[12];
ry(3.1387865188658415) q[11];
ry(-0.0020037083852688653) q[12];
cx q[11],q[12];
ry(-2.457454033417863) q[12];
ry(0.19524335184701072) q[13];
cx q[12],q[13];
ry(-1.380807431092208) q[12];
ry(-1.3532694133798309) q[13];
cx q[12],q[13];
ry(-2.3455392338768775) q[13];
ry(1.950756958674741) q[14];
cx q[13],q[14];
ry(3.121737301417063) q[13];
ry(0.37799672436794995) q[14];
cx q[13],q[14];
ry(-1.2628323491642428) q[14];
ry(-2.9703352359457407) q[15];
cx q[14],q[15];
ry(0.006465876141097837) q[14];
ry(-3.1340843132353235) q[15];
cx q[14],q[15];
ry(-3.130747506364473) q[15];
ry(1.0102329797046394) q[16];
cx q[15],q[16];
ry(-1.54961295504265) q[15];
ry(1.8689212647775015) q[16];
cx q[15],q[16];
ry(-1.8490176154420128) q[16];
ry(1.8082916187061646) q[17];
cx q[16],q[17];
ry(3.1144668549522043) q[16];
ry(3.138770138986521) q[17];
cx q[16],q[17];
ry(0.20477228406515113) q[17];
ry(-1.6251035724676202) q[18];
cx q[17],q[18];
ry(-1.0797680130977967) q[17];
ry(2.420334431991203) q[18];
cx q[17],q[18];
ry(-1.4298762942667462) q[18];
ry(0.019595112174906504) q[19];
cx q[18],q[19];
ry(0.3986991440068867) q[18];
ry(1.3164323131199271) q[19];
cx q[18],q[19];
ry(-1.2096303076174453) q[0];
ry(-1.359317452886244) q[1];
cx q[0],q[1];
ry(-2.4720704722436295) q[0];
ry(-0.08026914659361181) q[1];
cx q[0],q[1];
ry(-1.7137623592351554) q[1];
ry(-0.3557094121839386) q[2];
cx q[1],q[2];
ry(1.5319221273635062) q[1];
ry(2.944670751870445) q[2];
cx q[1],q[2];
ry(3.109811329733762) q[2];
ry(0.47997422649468646) q[3];
cx q[2],q[3];
ry(-3.138874092791517) q[2];
ry(-1.6921246427542664) q[3];
cx q[2],q[3];
ry(-1.134436338428759) q[3];
ry(-1.6089007924550982) q[4];
cx q[3],q[4];
ry(1.5963565073354735) q[3];
ry(0.00602114081295646) q[4];
cx q[3],q[4];
ry(3.130679825629898) q[4];
ry(-0.3819563846106711) q[5];
cx q[4],q[5];
ry(-0.797538021226351) q[4];
ry(1.5707816115980462) q[5];
cx q[4],q[5];
ry(2.291942739209968) q[5];
ry(-1.3650124847021567) q[6];
cx q[5],q[6];
ry(0.006686773964015051) q[5];
ry(-3.1396708632387447) q[6];
cx q[5],q[6];
ry(-2.791585980183634) q[6];
ry(-0.5911149118921655) q[7];
cx q[6],q[7];
ry(3.1360544250226687) q[6];
ry(-1.5701290762477402) q[7];
cx q[6],q[7];
ry(-2.1717668663676744) q[7];
ry(-0.26688147666969986) q[8];
cx q[7],q[8];
ry(-0.05222770176385705) q[7];
ry(1.5631606189680072) q[8];
cx q[7],q[8];
ry(-2.0046031580207266) q[8];
ry(-0.208106714249221) q[9];
cx q[8],q[9];
ry(1.5784308393755804) q[8];
ry(1.5980335067370603) q[9];
cx q[8],q[9];
ry(-2.5279269219285583) q[9];
ry(2.6353801177831664) q[10];
cx q[9],q[10];
ry(3.065877918275115) q[9];
ry(-0.013687127787169672) q[10];
cx q[9],q[10];
ry(1.7362297589036066) q[10];
ry(-1.589472991435712) q[11];
cx q[10],q[11];
ry(1.8503378685468557) q[10];
ry(-1.4438812881388363) q[11];
cx q[10],q[11];
ry(-0.7659465572412927) q[11];
ry(2.3734060483912915) q[12];
cx q[11],q[12];
ry(-3.1242712083333504) q[11];
ry(-3.138726282302949) q[12];
cx q[11],q[12];
ry(-1.3040628320027732) q[12];
ry(1.5335774527903556) q[13];
cx q[12],q[13];
ry(2.868877591941374) q[12];
ry(-0.004795025908947537) q[13];
cx q[12],q[13];
ry(-1.6059034336014526) q[13];
ry(1.0951407620142157) q[14];
cx q[13],q[14];
ry(3.1372865311334754) q[13];
ry(-2.7603169457857275) q[14];
cx q[13],q[14];
ry(1.1422707004627268) q[14];
ry(-1.6108845774802472) q[15];
cx q[14],q[15];
ry(3.1191882155260173) q[14];
ry(-3.117496199815079) q[15];
cx q[14],q[15];
ry(-0.10287006445758619) q[15];
ry(-0.2060855676708206) q[16];
cx q[15],q[16];
ry(2.5705000091121923) q[15];
ry(1.4018529981348538) q[16];
cx q[15],q[16];
ry(2.4386905240249503) q[16];
ry(-0.027452450075480604) q[17];
cx q[16],q[17];
ry(-3.137184592202666) q[16];
ry(3.139102349861717) q[17];
cx q[16],q[17];
ry(-0.03643560697972781) q[17];
ry(-1.436488180739804) q[18];
cx q[17],q[18];
ry(-0.021413417298700332) q[17];
ry(-1.409903969818361) q[18];
cx q[17],q[18];
ry(2.9728326609635642) q[18];
ry(0.02390637212627676) q[19];
cx q[18],q[19];
ry(-3.121072890995235) q[18];
ry(-1.5803786792009626) q[19];
cx q[18],q[19];
ry(-1.4244403258749) q[0];
ry(0.1616078911373977) q[1];
ry(1.7615465779613482) q[2];
ry(2.495287439926382) q[3];
ry(-0.7609862608374509) q[4];
ry(1.742492632565066) q[5];
ry(2.003212744984431) q[6];
ry(2.9604187260646717) q[7];
ry(-2.643643822195322) q[8];
ry(0.24758692814060268) q[9];
ry(2.0161481945998343) q[10];
ry(2.964687157103771) q[11];
ry(0.09904749201252105) q[12];
ry(-2.0136536733385313) q[13];
ry(-1.042268054921673) q[14];
ry(0.1417853020173263) q[15];
ry(-0.22458445902993648) q[16];
ry(1.2216125208099582) q[17];
ry(-0.5471899373844922) q[18];
ry(-1.7683426598892762) q[19];