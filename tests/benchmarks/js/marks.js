// Copyright 2002-2012, University of Colorado

/**
 * Compares performance benchmarks across version snapshots to help determine
 * improvements or regressions.
 *
 * @author Jonathan Olson
 */

// 'namespace'
var marks = marks || {};

(function(){
    
    marks.Timer = function( options ) {
        // onFinish()
        // onSnapshotAdd( snapshot )
        // onSnapshotStart( snapshot )
        // onSnapshotEnd( snapshot )
        // onBenchmarkComplete( snapshot, benchmark )
        this.options = options || {};
        
        this.markNames = []; // names for every benchmark run
        this.currentMarkNames = []; // names for the benchmarks run for the 'current' version
        
        this.snapshots = [];
        this.pendingSnapshots = [];
        
        this.running = false;
        
        this.currentSnapshot = null;
        
        // currently pulled from snapshot test code to add benchmarks
        this.currentSuite = null;
    };
    var Timer = marks.Timer;
    
    Timer.prototype = {
        constructor: Timer,
        
        addSnapshot: function( snapshot ) {
            this.snapshots.push( snapshot );
            this.pendingSnapshots.push( snapshot );
            
            if ( this.options.onSnapshotAdd ) {
                this.options.onSnapshotAdd( snapshot );
            }
        },
        
        compareSnapshots: function( snapshots ) {
            for ( var i = 0; i < snapshots.length; i++ ) {
                this.addSnapshot( snapshots[i] );
            }
            
            if ( !this.running ) {
                this.nextSnapshot();
            }
        },
        
        nextSnapshot: function() {
            var that = this;
            
            if ( this.pendingSnapshots.length !== 0 ) {
                var snapshot = this.pendingSnapshots.shift();
                this.currentSnapshot = snapshot;
                
                this.running = true;
                
                this.currentSuite = this.createSuite( snapshot );
                
                if( this.options.onSnapshotStart ) {
                    this.options.onSnapshotStart( snapshot );
                }
                
                // run library dependencies before running the tests
                var scripts = snapshot.libs.concat( snapshot.tests );
                
                function nextScript() {
                    if( scripts.length !== 0 ) {
                        var script = scripts.shift();
                        
                        loadScript( script, function() {
                            // once this script completes execution, run the next script
                            nextScript();
                        } );
                    } else {
                        // all scripts have executed
                        
                        // TODO: determine whether queued is necessary
                        that.currentSuite.run( { queued: true } );
                    }
                }
                
                nextScript();
            } else {
                this.currentSnapshot = null;
                
                if ( this.options.onFinish ) {
                    this.options.onFinish();
                }
            }
        },
        
        onBenchmarkComplete: function( event ) {
            var benchmark = event.target;
            
            if( this.options.onBenchmarkComplete ) {
                this.options.onBenchmarkComplete( this.currentSnapshot, benchmark );
            }
        },
        
        onSuiteComplete: function( event ) {
            var suite = this.currentSuite;
            
            if( this.options.onSnapshotEnd ) {
                this.options.onSnapshotEnd( this.currentSnapshot );
            }
            
            this.nextSnapshot();
        },
        
        createSuite: function( snapshot ) {
            var that = this;
            return new Benchmark.Suite( snapshot.name, {
                // TODO: strip out what we don't need from here
                onStart: function( event ) {
                    // console.log( snapshot.name + ' (suite onStart)' );
                    // console.log( event );
                },
                
                onCycle: function( event ) {
                    // console.log( snapshot.name + ' (suite onCycle)' );
                    // console.log( event );
                    that.onBenchmarkComplete( event );
                },
                
                onAbort: function( event ) {
                    // console.log( snapshot.name + ' (suite onAbort)' );
                    // console.log( event );
                },
                
                onError: function( event ) {
                    // console.log( snapshot.name + ' (suite onError)' );
                    // console.log( event );
                },
                
                onReset: function( event ) {
                    // console.log( snapshot.name + ' (suite onReset)' );
                    // console.log( event );
                },
                
                onComplete: function( event ) {
                    // console.log( snapshot.name + ' (suite onComplete)' );
                    // console.log( event );
                    that.onSuiteComplete( event );
                }
            } );
        },
        
        run: function( benchmarks ) {
            
        },
        
        display: function() {
            
        }
    }
    
    function debug( msg ) {
        if ( console && console.log ) {
            console.log( msg );
        }
    }
    
    function info( msg ) {
        if ( console && console.log ) {
            console.log( msg );
        }
    }
    
    function loadScript( src, callback ) {
        debug( 'requesting script ' + src );
        
        var called = false;
        
        var script = document.createElement( 'script' );
        script.type = 'text/javascript';
        script.async = true;
        script.onload = script.onreadystatechange = function() {
            var state = this.readyState;
            if ( state && state != "complete" && state != "loaded" ) {
                return;
            }
            
            if ( !called ) {
                debug( 'completed script ' + src );
                called = true;
                callback();
            }
        };
        
        // make sure things aren't cached, just in case
        script.src = src + '?random=' + Math.random().toFixed( 10 );
        
        var other = document.getElementsByTagName( 'script' )[0];
        other.parentNode.insertBefore( script, other );
    }
    
})();
