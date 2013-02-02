
(function(){
    
    var results = $( '#results' );
    
    var currentRunCount = 0;
    
    var data = {};
    
    // stores all of the tests that will run with the current version, so that we can compare against older versions
    var currentNames = [];
    
    var versionNames = [];
    
    window.sceneBench = function( name, fn ) {
        var version = window.currentTestVersionName;
        
        console.log( 'loading test ' + name + ' (' + version + ')' );
        
        if( version === 'current' ) {
            currentNames.push( name );
        }
        
        currentRunCount = currentRunCount + 1;
        
        var bench = new Benchmark( name, fn, {
            defer: true,
            
            minTime: 20,
            
            onCycle: function( event ) {
                $( '#main' ).empty();
            },
            
            onComplete: function( event ) {
                console.log( name + ' (' + version + ') complete' );
                console.log( event );
                
                data[version][name] = event.target;
                
                currentRunCount = currentRunCount - 1;
                
                if( currentRunCount == 0 ) {
                    nextVersion();
                }
            }
        } );
        
        bench.run( {
            async: true
        } );
    };
    
    function loadScript( src, callback ) {
        console.log( 'loading script ' + src );
        
        var called = false;
        
        var script = document.createElement( 'script' );
        script.type = 'text/javascript';
        script.async = true;
        script.onload = script.onreadystatechange = function() {
            var state = this.readyState;
            if ( state && state != "complete" && state != "loaded" ) {
                return;
            }
            
            if( !called ) {
                console.log( 'loaded script ' + src );
                called = true;
                callback();
            }
        };
        script.src = src;
        
        var other = document.getElementsByTagName( 'script' )[0];
        other.parentNode.insertBefore( script, other );
    }
    
    var versionsRemaining = [
        { name: 'control', lib: '../../phet-scene-min.js', test: 'js/current.js' },
        { name: '120202-515bfa9d9a', lib: 'snapshots/phet-scene-min-120202-515bfa9d9a.js', test: 'snapshots/tests-120202-515bfa9d9a.js' },
        { name: 'current', lib: '../../phet-scene-min.js', test: 'js/current.js' }
    ];
    
    function progressBar( currentHz, testHz ) {
        var td = $( document.createElement( 'td' ) );
        var extraClass = currentHz < testHz ? 'bar-danger' : 'bar-success';
        var maxHz = Math.max( currentHz, testHz );
        var minHz = Math.min( currentHz, testHz );
        
        var baseWidth = Math.round( 100 * minHz / maxHz );
        
        var baseBar = '<div class="bar" style="width: ' + baseWidth.toFixed() + '%;"></div>'
        var extraBar = '<div class="bar ' + extraClass + '" style="width: ' + ( 100 - baseWidth ).toFixed() + '%;"></div>'
        
        td.html( '<div class="progress">' + baseBar + extraBar + '</div>' );
        return td;
    }
    
    function nextVersion() {
        if( versionsRemaining.length > 0 ) {
            var version = versionsRemaining.pop();
            
            if( version.name !== 'current' ) {
                versionNames.push( version.name );
            }
            window.currentTestVersionName = version.name;
            
            console.log( 'running ' + version.test );
            
            data[version.name] = {};
            
            loadScript( version.lib, function() {
                loadScript( version.test, function() {} );
            } );
        } else {
            // all done
            
            _.each( versionNames, function( versionName ) {
                var tableHeader = $( document.createElement( 'th' ) ).text( versionName );
                $( '#versions' ).append( tableHeader );
                $( '#versions' ).append( document.createElement( 'th' ) );
                $( '#versions' ).append( document.createElement( 'th' ) );
            } );
            
            _.each( currentNames, function( name ) {
                var tr = $( document.createElement( 'tr' ) );
                tr.addClass( 'result' );
                tr.html( '<td class="name">' + name + '</td>' );
                
                var currentHz = data.current[name].hz;
                
                _.each( versionNames, function( versionName ) {
                    var test = data[versionName][name];
                    if( test ) {
                        var testHz = test.hz;
                        tr.append( progressBar( currentHz, testHz ) );
                        tr.append( $( document.createElement( 'td' ) ).text( ( testHz / currentHz ).toFixed( 3 ) ) );
                        tr.append( $( document.createElement( 'td' ) ).text( testHz.toFixed( 1 ) ) );
                    } else {
                        tr.append( document.createElement( 'td' ) );
                        tr.append( document.createElement( 'td' ) );
                        tr.append( document.createElement( 'td' ) );
                    }
                } );
                
                results.append( tr );
            } );
        }
    }
    
    nextVersion();
    
    // loadScript( '../../phet-scene-min.js', function() {
    //     console.log( new phet.scene.Node() );
    //     loadScript( 'js/current.js', function() {} );
    // } );
})();
