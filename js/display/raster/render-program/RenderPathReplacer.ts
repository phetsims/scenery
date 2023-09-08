// Copyright 2023, University of Colorado Boulder

/**
 * A optimized form for constructing RenderPrograms with their RenderPathBooleans simplified.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderDepthSort, RenderPath, RenderPathBoolean, RenderPlanar, RenderProgram, RenderStack, RenderTrail, scenery } from '../../../imports.js';

const emptyIndexArray: number[] = [];

export default class RenderPathReplacer {

  // For a given RenderPath, stores a trail for every place it is found within the RenderProgram
  private readonly trailsMap = new Map<RenderPath, RenderTrail[]>();

  // For a given container (that has a RenderPathBoolean as a descendant), stores the indices of its children that are
  // permanent (not a transparent-outside RenderPathBoolean)
  private readonly permanentChildrenMap = new Map<RenderStack | RenderDepthSort, number[]>();

  public constructor(
    public readonly program: RenderProgram
  ) {
    assert && assert( program.isSimplified, 'Should have simplified it by now' );

    this.initialRecurse( program, [] );
  }

  public replace( includedPaths: Set<RenderPath> ): RenderProgram {
    if ( !this.program.hasPathBoolean ) {
      return this.program;
    }

    const trails: RenderTrail[] = [];
    for ( const path of includedPaths ) {
      trails.push( ...this.trailsMap.get( path ) ?? [] );
    }

    trails.sort( RenderTrail.closureCompare );

    const scaffold = new RenderScaffold( -1 );
    for ( let i = 0; i < trails.length; i++ ) {
      scaffold.add( trails[ i ].indices, 0 );
    }

    return this.replaceRecurse( this.program, includedPaths, scaffold ).simplified();
  }

  private replaceRecurse( program: RenderProgram, includedPaths: Set<RenderPath>, scaffold: RenderScaffold | null ): RenderProgram {
    if ( !program.hasPathBoolean ) {
      return program;
    }

    if ( program instanceof RenderPathBoolean ) {
      // TODO: should we have "isIncluded" as part of the scaffold?
      // const included = scaffold ? scaffold.isIncluded : includedPaths.has( program.path );
      const included = includedPaths.has( program.path );

      return this.replaceRecurse(
        included ? program.inside : program.outside,
        includedPaths,
        scaffold ? scaffold.getAtIndex( included ? 0 : 1 ) : null
      );
    }
    else if ( program instanceof RenderStack || program instanceof RenderDepthSort ) {
      const permanentChildrenIndices = this.permanentChildrenMap.get( program ) || emptyIndexArray;
      const pathIncludedIndices = scaffold ? scaffold.children.filter( s => s.isIncluded ).map( s => s.index ) : emptyIndexArray;
      const reversedChildren: RenderProgram[] = [];
      const reversedItems: RenderPlanar[] = [];

      let permanentIndexIndex = permanentChildrenIndices.length - 1;
      let pathIndexIndex = pathIncludedIndices.length - 1;
      const isStack = program instanceof RenderStack;

      while ( permanentIndexIndex >= 0 || pathIndexIndex >= 0 ) {
        const permanentIndex = permanentIndexIndex >= 0 ? permanentChildrenIndices[ permanentIndexIndex ] : -1;
        const pathIndex = pathIndexIndex >= 0 ? pathIncludedIndices[ pathIndexIndex ] : -1;

        let index;
        if ( permanentIndex > pathIndex ) {
          index = permanentIndex;
          permanentIndexIndex--;
        }
        else {
          index = pathIndex;
          pathIndexIndex--;
        }

        const child = this.replaceRecurse( program.children[ index ], includedPaths, scaffold ? scaffold.getAtIndex( index ) : null );

        if ( isStack ) {
          reversedChildren.push( child );
        }
        else {
          reversedItems.push( program.items[ index ].withProgram( child ) );
        }

        if ( isStack && child.isFullyOpaque ) {
          break;
        }
      }

      if ( isStack ) {
        return new RenderStack( reversedChildren.reverse() );
      }
      else {
        return new RenderDepthSort( reversedItems.reverse() );
      }
    }
    else {
      return program.withChildren( program.children.map( ( child, i ) => {
        return this.replaceRecurse( child, includedPaths, scaffold ? scaffold.getAtIndex( i ) : null );
      } ) );
    }
  }

  private initialRecurse( program: RenderProgram, indices: number[] ): void {
    // If there are no path-booleans in this subtree, skip it (we'll be constructing it directly anyway).
    if ( !program.hasPathBoolean ) {
      return;
    }

    if ( RenderPathReplacer.isTransparentOutside( program ) ) {
      this.addTrail( program.path, new RenderTrail( this.program, indices.slice() ) );
    }

    // We'll want to check to see if we've traversed and encountered this container before (if so, we need to skip it,
    // so we don't store duplicate indices).
    const addPermanentChildren = RenderPathReplacer.isContainer( program ) && !this.permanentChildrenMap.has( program );

    for ( let i = 0; i < program.children.length; i++ ) {
      indices.push( i );

      const child = program.children[ i ];

      this.initialRecurse( child, indices );

      if ( addPermanentChildren && !RenderPathReplacer.isTransparentOutside( child ) ) {
        this.addPermanentChild( program, i );
      }

      indices.pop();
    }
  }

  private addTrail( program: RenderPath, trail: RenderTrail ): void {
    let trails = this.trailsMap.get( program );
    if ( !trails ) {
      trails = [];
      this.trailsMap.set( program, trails );
    }
    trails.push( trail );
  }

  private addPermanentChild( program: RenderStack | RenderDepthSort, index: number ): void {
    let indices = this.permanentChildrenMap.get( program );
    if ( !indices ) {
      indices = [];
      this.permanentChildrenMap.set( program, indices );
    }
    indices.push( index );
  }

  public static isContainer( program: RenderProgram ): program is ( RenderStack | RenderDepthSort ) {
    return program instanceof RenderStack || program instanceof RenderDepthSort;
  }

  public static isTransparentOutside( program: RenderProgram ): program is RenderPathBoolean {
    return program instanceof RenderPathBoolean && program.outside.isFullyTransparent;
  }
}

scenery.register( 'RenderPathReplacer', RenderPathReplacer );

class RenderScaffold {

  public readonly children: RenderScaffold[] = [];
  public isIncluded = false;

  public constructor(
    public readonly index: number
  ) {}

  public add( indices: number[], startingIndex: number ): void {
    if ( startingIndex === indices.length ) {
      this.isIncluded = true;
    }
    else {
      if ( this.children.length === 0 ) {
        const scaffold = new RenderScaffold( indices[ startingIndex ] );
        this.children.push( scaffold );
        scaffold.add( indices, startingIndex + 1 );
      }
      else {
        const lastChild = this.children[ this.children.length - 1 ];

        if ( lastChild.index === indices[ startingIndex ] ) {
          lastChild.add( indices, startingIndex + 1 );
        }
        else {
          assert && assert( lastChild.index < indices[ startingIndex ], 'These should be executed in-order' );
          const scaffold = new RenderScaffold( indices[ startingIndex ] );
          this.children.push( scaffold );
          scaffold.add( indices, startingIndex + 1 );
        }
      }
    }
  }

  public getAtIndex( index: number ): RenderScaffold | null {
    for ( let i = 0; i < this.children.length; i++ ) {
      const child = this.children[ i ];
      if ( child.index === index ) {
        return child;
      }
      // We are in order!
      else if ( child.index > index ) {
        return null;
      }
    }
    return null;
  }
}
