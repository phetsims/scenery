// Copyright 2016, University of Colorado Boulder

var marks = marks || {};

(function() {
  'use strict';
  
  marks.TableBase = function( container ) {
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

    this.numRows = 0;
    this.numColumns = 0;

    this.benchmarkRowNumbers = {}; // indexed by benchmark name
    this.snapshotColumnNumbers = {}; // indexed by snapshot
  };
  var TableBase = marks.TableBase;

  TableBase.prototype = {
    constructor: TableBase,

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

      return this.numRows - 1;
    },

    addColumn: function( name, colSpan ) {
      var numColumns = ( colSpan && colSpan > 1 ) ? colSpan : 1;

      var header = document.createElement( 'th' );
      header.appendChild( document.createTextNode( name ) );
      if ( numColumns > 1 ) {
        header.colSpan = colSpan;
      }
      this.headRow.appendChild( header );

      for ( var k = 0; k < numColumns; k++ ) {
        this.numColumns++;

        // append column to each row
        for ( var i = 0; i < this.numRows; i++ ) {
          var td = document.createElement( 'td' );
          this.rows[ i ].appendChild( td );
          this.cells[ i ].push( td );
        }
      }

      return this.numColumns - numColumns;
    }
  };
})();
