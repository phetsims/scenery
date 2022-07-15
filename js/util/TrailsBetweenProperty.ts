// Copyright 2022, University of Colorado Boulder

/**
 * A Property that will synchronously contain all Trails between two nodes (in a root-leaf direction).
 * Listens from the child to the parent (since we tend to branch much less that way).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import { Node, scenery, Trail } from '../imports.js';

export default class TrailsBetweenProperty extends TinyProperty<Trail[]> {

  public readonly rootNode: Node;
  public readonly leafNode: Node;
  public readonly listenedNodeSet: Set<Node> = new Set<Node>();
  private readonly _trailUpdateListener: () => void;

  public constructor( rootNode: Node, leafNode: Node ) {
    super( [] );

    this.rootNode = rootNode;
    this.leafNode = leafNode;

    this._trailUpdateListener = this.update.bind( this );

    this.update();
  }

  private update(): void {
    // Trails accumulated in our recursion that will be our Property's value
    const trails: Trail[] = [];

    // Nodes that were touched in the scan (we should listen to changes to ANY of these to see if there is a connection
    // or disconnection. This could potentially cause our Property to change
    const nodeSet = new Set<Node>();

    // Modified in-place during the search
    const trail = new Trail( this.leafNode );

    const rootNode = this.rootNode;
    ( function recurse() {
      const root = trail.rootNode();

      nodeSet.add( root );

      if ( root === rootNode ) {
        // Create a permanent copy that won't be mutated
        trails.push( trail.copy() );
      }

      root.parents.forEach( parent => {
        trail.addAncestor( parent );
        recurse();
        trail.removeAncestor();
      } );
    } )();

    // Add in new needed listeners
    nodeSet.forEach( node => {
      if ( !this.listenedNodeSet.has( node ) ) {
        this.addNodeListener( node );
      }
    } );

    // Remove listeners not needed anymore
    this.listenedNodeSet.forEach( node => {
      if ( !nodeSet.has( node ) ) {
        this.removeNodeListener( node );
      }
    } );

    // Guard in a way that deepEquality on the Property wouldn't (because of the Array wrapper)
    const currentTrails = this.value;
    let trailsEqual = currentTrails.length === trails.length;
    if ( trailsEqual ) {
      for ( let i = 0; i < trails.length; i++ ) {
        if ( !currentTrails[ i ].equals( trails[ i ] ) ) {
          trailsEqual = false;
          break;
        }
      }
    }

    if ( !trailsEqual ) {
      this.value = trails;
    }
  }

  private addNodeListener( node: Node ): void {
    this.listenedNodeSet.add( node );
    node.parentAddedEmitter.addListener( this._trailUpdateListener );
    node.parentRemovedEmitter.addListener( this._trailUpdateListener );
  }

  private removeNodeListener( node: Node ): void {
    this.listenedNodeSet.delete( node );
    node.parentAddedEmitter.removeListener( this._trailUpdateListener );
    node.parentRemovedEmitter.removeListener( this._trailUpdateListener );
  }

  public override dispose(): void {
    this.listenedNodeSet.forEach( node => this.removeNodeListener( node ) );

    super.dispose();
  }
}

scenery.register( 'TrailsBetweenProperty', TrailsBetweenProperty );
