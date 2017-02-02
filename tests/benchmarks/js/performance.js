// Copyright 2016, University of Colorado Boulder

var marks = marks || {};

(function() {
  'use strict';

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
      var self = this;

      // guard on markNames so we don't trample over the first run
      if ( this.markNames && this.snapshotIndex === this.snapshots.length ) {
        this.snapshotIndex = 0;
        this.markNameIndex++;

        if ( this.markNameIndex === this.markNames.length ) {
          this.markNameIndex = 0;
        }
      }

      var snapshot = this.snapshots[ this.snapshotIndex++ ];
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
        }
        else {
          // all scripts have executed

          if ( !self.markNames ) {
            self.markNames = _.map( marks.currentMarks, function( mark ) { return mark.name; } );
          }

          setTimeout( function() {
            self.runMark();
          }, 250 );
        }
      }

      nextScript();
    },

    runMark: function() {
      var self = this;

      var mark = this.findMark( this.markNames[ this.markNameIndex ] );

      if ( mark ) {
        mark.before && mark.before();

        var count = 0;

        if ( !mark.count ) {
          mark.count = 50;
        }

        var tick = function() {
          if ( count++ === mark.count ) {
            var time = new Date - startTime;
            if ( self.options.onMark ) {
              self.options.onMark( self.currentSnapshot, mark, time / mark.count, self );
            }
            mark.after && mark.after();

            setTimeout( function() {
              self.runSnapshots();
            }, 500 );
          }
          else {
            window.requestAnimationFrame( tick, main[ 0 ] ); // eslint-disable-line no-undef

            mark.step && mark.step();
          }
        };

        window.requestAnimationFrame( tick, main[ 0 ] ); // eslint-disable-line no-undef

        var startTime = new Date;
      }
      else {
        // skip this one
        self.runSnapshots();
      }
    }
  };

  // passed to marks.Performance constructor as an options object. container is a block-level element that the report is placed in
  marks.PerformanceTableReport = function( container ) {
    this.table = new marks.TableBase( container );

    this.initialized = false;

    this.snapshotColumnMap = {};
    this.markNameRowMap = {};
  };
  var PerformanceTableReport = marks.PerformanceTableReport;

  PerformanceTableReport.prototype = {
    constructor: PerformanceTableReport,

    initializeTable: function( performance ) {
      var self = this;
      this.initialized = true;

      this.table.addColumn( 'Benchmark Name' );

      _.each( performance.snapshots, function( snapshot ) {
        self.snapshotColumnMap[ snapshot.name ] = self.table.numColumns;
        self.table.addColumn( snapshot.name, 2 );
      } );

      _.each( performance.markNames, function( markName ) {
        var rowNumber = self.table.addRow();
        self.markNameRowMap[ markName ] = rowNumber;
        self.table.cells[ rowNumber ][ 0 ].innerHTML = markName;
      } );
    },

    onMark: function( snapshot, mark, ms, performance ) {
      if ( !this.initialized ) {
        this.initializeTable( performance );
      }

      if ( !snapshot.times ) {
        snapshot.times = {};
      }
      if ( !snapshot.times[ mark.name ] ) {
        snapshot.times[ mark.name ] = {
          total: 0,
          count: 0
        };
      }

      var time = snapshot.times[ mark.name ];
      time.total += ms;
      time.count += 1;

      var average = time.total / time.count;
      this.table.cells[ this.markNameRowMap[ mark.name ] ][ this.snapshotColumnMap[ snapshot.name ] ].innerHTML = average.toFixed( 1 );
      if ( snapshot.name !== 'current' ) {
        var currentTime = performance.snapshots[ 0 ].times[ mark.name ];
        var currentAverage = currentTime.total / currentTime.count;
        var percentageChange = 100 * ( currentAverage - average ) / average;
        this.table.cells[ this.markNameRowMap[ mark.name ] ][ this.snapshotColumnMap[ snapshot.name ] + 1 ].innerHTML = percentageChange.toFixed( 2 ) + '%';
      }
      // console.log( snapshot.name + ' ' + mark.name + ': ' + average );
    }
  };

  function debug( msg ) {
    if ( console && console.log ) {
      // console.log( msg );
    }
  }

  // function info( msg ) {
  //   if ( console && console.log ) {
  //     console.log( msg );
  //   }
  // }

  function loadScript( src, callback ) {
    debug( 'requesting script ' + src );

    var called = false;

    var script = document.createElement( 'script' );
    script.type = 'text/javascript';
    script.async = true;
    script.onload = script.onreadystatechange = function() {
      var state = this.readyState;
      if ( state && state !== 'complete' && state !== 'loaded' ) {
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

    var other = document.getElementsByTagName( 'script' )[ 0 ];
    other.parentNode.insertBefore( script, other );
  }

})();
