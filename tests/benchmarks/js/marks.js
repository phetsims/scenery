// Copyright 2002-2016, University of Colorado Boulder


/**
 * Compares performance benchmarks across version snapshots to help determine
 * improvements or regressions.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

// 'namespace'
var marks = marks || {};

(function() {
  'use strict';

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

    add: function( name, fn, args ) {
      args = args || {};

      if ( args.delay === undefined ) {
        args.delay = 0.1; // increased delay so hopefully we don't muck up things with our table displays, etc.
      }
      if ( args.minTime === undefined ) {
        // args.minTime = 1;
      }

      this.currentSuite.add( name, fn, args );
    },

    addSnapshot: function( snapshot ) {
      this.snapshots.push( snapshot );
      this.pendingSnapshots.push( snapshot );

      if ( this.options.onSnapshotAdd ) {
        this.options.onSnapshotAdd( snapshot );
      }
    },

    compareSnapshots: function( snapshots ) {
      for ( var i = 0; i < snapshots.length; i++ ) {
        this.addSnapshot( snapshots[ i ] );
      }

      if ( !this.running ) {
        this.nextSnapshot();
      }
    },

    nextSnapshot: function() {

      if ( this.pendingSnapshots.length !== 0 ) {
        var snapshot = this.pendingSnapshots.shift();
        this.currentSnapshot = snapshot;

        this.running = true;

        this.currentSuite = this.createSuite( snapshot );

        var suite = this.currentSuite;

        if ( this.options.onSnapshotStart ) {
          this.options.onSnapshotStart( snapshot );
        }

        // run library dependencies before running the tests
        var scripts = snapshot.libs.concat( snapshot.tests );

        var nextScript = function() {
          if ( scripts.length !== 0 ) {
            var script = scripts.shift();

            loadScript( script, function() {
              // once this script completes execution, run the next script
              nextScript();
            } );
          }
          else {
            // all scripts have executed

            // TODO: determine whether queued is necessary
            suite.run( { queued: true } );
          }
        };

        nextScript();
      }
      else {
        this.currentSnapshot = null;

        if ( this.options.onFinish ) {
          this.options.onFinish();
        }
      }
    },

    onBenchmarkComplete: function( event ) {
      var benchmark = event.target;

      if ( this.options.onBenchmarkComplete ) {
        this.options.onBenchmarkComplete( this.currentSnapshot, benchmark );
      }
    },

    onSuiteComplete: function( event ) {
      if ( this.options.onSnapshotEnd ) {
        this.options.onSnapshotEnd( this.currentSnapshot );
      }

      this.nextSnapshot();
    },

    createSuite: function( snapshot ) {
      var self = this;
      return new Benchmark.Suite( snapshot.name, { // eslint-disable-line no-undef
        // TODO: strip out what we don't need from here
        onStart: function( event ) {
          // console.log( snapshot.name + ' (suite onStart)' );
          // console.log( event );
        },

        onCycle: function( event ) {
          // console.log( snapshot.name + ' (suite onCycle)' );
          // console.log( event );
          self.onBenchmarkComplete( event );
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
          self.onSuiteComplete( event );
        }
      } );
    },

    run: function( benchmarks ) {

    },

    display: function() {

    }
  };

  // passed to marks.Timer constructor as an options object. container is a block-level element that the report is placed in
  marks.TableReport = function( container ) {
    this.container = container;

    this.table = document.createElement( 'table' );
    this.table.className = 'table table-condensed';

    this.thead = document.createElement( 'thead' );
    this.table.appendChild( this.thead );
    this.headRow = document.createElement( 'tr' );
    this.thead.appendChild( this.headRow );

    this.tbody = document.createElement( 'tbody' );
    this.table.appendChild( this.tbody );

    this.container.appendChild( this.table );

    // 2-d array that stores table cells (TD elements)
    this.cells = [];

    // TR elements
    this.rows = [];

    // 2-d array that is benchmarks[row][column]
    this.benchmarks = [];

    this.numRows = 0;
    this.numColumns = 0;

    this.benchmarkRowNumbers = {}; // indexed by benchmark name
    this.snapshotColumnNumbers = {}; // indexed by snapshot


    this.benchmarkNameColumn = this.numColumns;
    this.addColumn( 'Benchmark Name' );

    this.currentNameColumn = this.numColumns;
  };
  var TableReport = marks.TableReport;

  TableReport.prototype = {
    constructor: TableReport,

    addStats: function( snapshot, benchmark, cell ) {

      var showPlusMinus = false;

      function log10( x ) {
        return Math.LOG10E * Math.log( x );
      }

      var multiplier;
      var units;
      if ( benchmark.stats.mean >= 1 ) {
        multiplier = 1;
        units = 's';
      }
      else if ( benchmark.stats.mean >= 0.001 ) {
        multiplier = 1000;
        units = 'ms';
      }
      else if ( benchmark.stats.mean >= 0.000001 ) {
        multiplier = 1000000;
        units = '&#xb5;s';
      }
      else {
        multiplier = 1000000000;
        units = 'ns';
      }

      var ms = multiplier * benchmark.stats.mean;
      var moe = multiplier * benchmark.stats.moe;

      var digits = Math.min( 20, -Math.min( 0, Math.floor( log10( moe ) ) + ( showPlusMinus ? -1 : 1 ) ) ); // add another digit over significant figures
      if ( ms === 0 ) {
        digits = 0;
      }

      var html = ms.toFixed( digits ) + ( !showPlusMinus || moe === 0 ? '' : '&#xb1' + moe.toFixed( digits ) ) + units;

      if ( snapshot.name !== 'current' ) {
        var currentMark = this.benchmarks[ this.benchmarkRowNumbers[ benchmark.name ] ][ this.currentNameColumn ];

        var percentChange = ( -100 * ( currentMark.stats.mean - benchmark.stats.mean ) / benchmark.stats.mean ).toFixed( 2 );
        var meanDifference = Math.abs( currentMark.stats.mean - benchmark.stats.mean );
        var marginOfErrorAllowance = currentMark.stats.moe + benchmark.stats.moe;
        var worse = currentMark.stats.mean > benchmark.stats.mean;

        // TODO: put these in different columns?
        if ( meanDifference > marginOfErrorAllowance ) {
          if ( meanDifference > 10 * marginOfErrorAllowance ) {
            cell.style.background = worse ? '#ffcccc' : '#ccffcc';
          }
          else {
            cell.style.background = worse ? '#ffeeee' : '#eeffee';
          }
          html = '<strong>' + percentChange + '%</strong> ' + html;
        }
        else {
          cell.style.background = '#ffffff';
          html = percentChange + '% ' + html;
        }
      }

      cell.innerHTML = html;
    },

    addSnapshot: function( snapshot ) {
      this.snapshotColumnNumbers[ snapshot ] = this.numColumns;

      this.addColumn( snapshot.name );
    },

    addBenchmark: function( snapshot, benchmark ) {
      if ( !( benchmark.name in this.benchmarkRowNumbers ) ) {
        var rowNumber = this.numRows;
        this.benchmarkRowNumbers[ benchmark.name ] = rowNumber;
        this.benchmarks.push( [] );

        this.addRow();

        this.cells[ rowNumber ][ this.benchmarkNameColumn ].appendChild( document.createTextNode( benchmark.name ) );
      }

      var row = this.benchmarkRowNumbers[ benchmark.name ];
      var column = this.snapshotColumnNumbers[ snapshot ];

      this.benchmarks[ row ][ column ] = benchmark;

      this.addStats( snapshot, benchmark, this.cells[ row ][ column ] );
    },

    addRow: function() {
      this.numRows++;

      var row = document.createElement( 'tr' );
      this.table.appendChild( row );
      this.rows.push( row );

      // append row with the requisite number of columns
      var rowElements = [];
      for ( var i = 0; i < this.numColumns; i++ ) {
        var td = document.createElement( 'td' );
        row.appendChild( td );
        rowElements.push( td );
      }
      this.cells.push( rowElements );
    },

    addColumn: function( name ) {
      var header = document.createElement( 'th' );
      header.appendChild( document.createTextNode( name ) );
      this.headRow.appendChild( header );

      this.numColumns++;

      // append column to each row
      for ( var i = 0; i < this.numRows; i++ ) {
        var td = document.createElement( 'td' );
        this.rows[ i ].appendChild( td );
        this.cells[ i ].push( td );
      }
    },

    onFinish: function() {
      console.log( 'Timer.onFinish' );
    },

    onSnapshotAdd: function( snapshot ) {
      console.log( 'Timer.onSnapshotAdd: ' + snapshot.name );
    },

    onSnapshotStart: function( snapshot ) {
      console.log( 'Timer.onSnapshotStart: ' + snapshot.name );
      console.log( 'BOO' );
      this.addSnapshot( snapshot );
    },

    onSnapshotEnd: function( snapshot ) {
      console.log( 'Timer.onSnapshotEnd: ' + snapshot.name );
    },

    onBenchmarkComplete: function( snapshot, benchmark ) {
      console.log( 'Timer.onBenchmarkComplete: ' + snapshot.name + ' ' + benchmark.name );
      console.log( benchmark );
      this.addBenchmark( snapshot, benchmark );
    }
  };

  function debug( msg ) {
    if ( console && console.log ) {
      console.log( msg );
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
