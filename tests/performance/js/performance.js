
var marks = marks || {};

(function(){
  
  marks.Performance = function( options ) {
    this.options = options || {};
  };
  var Performance = marks.Performance;
  
  Performance.prototype = {
    constructor: Performance,
    
    compareSnapshots: function( snapshots ) {
      this.snapshots = snapshots;
      this.currentSnapshot = null;
      this.markNames = null;
      this.snapshotIndex = 0;
      this.markNameIndex = 0;
      
      this.runSnapshots();
    },
    
    findMark: function( name ) {
      return _.find( marks.currentMarks, function( mark ) { return mark.name === name; } );
    },
    
    runSnapshots: function() {
      // TODO: load in other direction?
      var that = this;
      
      // guard on markNames so we don't trample over the first run
      if( this.markNames && this.snapshotIndex === this.snapshots.length ) {
        this.snapshotIndex = 0;
        this.markNameIndex++;
        
        if( this.markNameIndex === this.markNames.length ) {
          this.markNameIndex = 0;
        }
      }
      
      var snapshot = this.snapshots[this.snapshotIndex++];
      this.currentSnapshot = snapshot;
      
      // run library dependencies before running the tests
      var scripts = snapshot.libs.concat( snapshot.tests );
      
      function nextScript() {
        if ( scripts.length !== 0 ) {
          var script = scripts.shift();
          
          loadScript( script, function() {
            // once this script completes execution, run the next script
            nextScript();
          } );
        } else {
          // all scripts have executed
          
          if( !that.markNames ) {
            that.markNames = _.map( marks.currentMarks, function( mark ) { return mark.name; } );
          }
          
          that.runMark();
        }
      }
      
      nextScript();
    },
    
    runMark: function() {
      var that = this;
      
      var mark = this.findMark( this.markNames[this.markNameIndex] );
      
      mark.before && mark.before();
      
      var count = 0;
      
      if( !mark.count ) {
        mark.count = 50;
      }
      
      function tick() {
        if( count++ === mark.count ) {
          var time = new Date - startTime;
          if( that.options.onMark ) {
            that.options.onMark( that.currentSnapshot, mark, time / mark.count );
          }
          mark.after && mark.after();
          
          that.runSnapshots();
        } else {
          window.requestAnimationFrame( tick, main[0] );
          
          mark.step && mark.step();
        }
      }
      window.requestAnimationFrame( tick, main[0] );
      
      var startTime = new Date;
    }
  };
  
  function debug( msg ) {
    if ( console && console.log ) {
      // console.log( msg );
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
